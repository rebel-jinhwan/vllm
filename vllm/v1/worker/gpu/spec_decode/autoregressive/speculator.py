# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.dp_utils import DPSyncState, dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
    SpeculatorCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.kernels import ModelRunnerKernels
from vllm.v1.worker.utils import AttentionGroup, get_uniform_decode_token_count

logger = init_logger(__name__)


class AutoRegressiveSpeculator(DraftModelSpeculator):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        kernels: ModelRunnerKernels,
    ):
        super().__init__(vllm_config, device, kernels)

        self.hidden_states = torch.zeros(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
        )
        self.current_draft_step = torch.tensor(0, dtype=torch.int64, device=device)
        self.last_token_indices = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        self.sample_src_positions = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )

        self.inputs_embeds: torch.Tensor | None = None

        self.prefill_cudagraph_manager: SpeculatorCudaGraphManager | None = None
        self.decode_cudagraph_manager: SpeculatorCudaGraphManager | None = None
        self.use_fused_multi_step_decode = False

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        if not self.supports_mm_inputs:
            return

        self.inputs_embeds = torch.zeros(
            self.max_num_tokens,
            self.hidden_size,
            dtype=self.dtype,
            device=self.device,
        )

    # Lifecycle hooks for model-specific optimizations. Subclasses override
    # the ones they need. These fire in both `capture` and `propose` so that
    # any state they toggle (e.g. attention flags baked into a CUDA graph) is
    # identical at capture time and replay time.
    def on_prefill_begin(self, num_reqs: int) -> None: ...

    def on_prefill_end(self, num_reqs: int) -> None: ...

    def on_multi_step_decode_begin(self, num_reqs: int) -> None: ...

    def on_multi_step_decode_end(self, num_reqs: int) -> None: ...

    @property
    def advance_draft_positions(self) -> bool:
        """Whether to increment positions and seq_lens between draft steps.

        True for Eagle/standard MTP (each step produces new KV).
        False for Gemma4 MTP (Q-only, shares target KV, constant positions).
        """
        return True

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers: InputBuffers,
        target_attn_groups: list[list[AttentionGroup]],
    ) -> None:
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        self._configure_fused_multi_step_decode()

    def _configure_fused_multi_step_decode(self) -> None:
        if self.num_speculative_steps == 1:
            self.use_fused_multi_step_decode = False
            return

        if not self.advance_draft_positions:
            self.use_fused_multi_step_decode = True
            return

        unsupported_backends = sorted(
            {
                attn_group.backend.get_name()
                for attn_groups in self.attn_groups
                for attn_group in attn_groups
                if not attn_group.supports_draft_decode_metadata_update
            }
        )
        self.use_fused_multi_step_decode = not unsupported_backends
        if unsupported_backends:
            logger.info_once(
                "Fused multi-step draft decode is not supported by attention "
                "backend(s) %s; falling back to rebuilding attention metadata "
                "between draft steps.",
                ", ".join(unsupported_backends),
            )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # Initialize cudagraph manager for draft prefill (draft position 0).
        self.prefill_cudagraph_manager = SpeculatorCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            self.num_speculative_steps + 1,
        )

        # PIECEWISE cudagraphs are not supported for draft decodes.
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            cudagraph_mode = CUDAGraphMode.NONE

        # Initialize cudagraph manager for draft decodes (draft positions > 0).
        self.decode_cudagraph_manager = SpeculatorCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_len=1,
        )

    def capture(self) -> None:
        logger.info("Capturing model for speculator...")
        # Reset indices to zeros to prevent stale values from prior
        # dummy runs to cause out-of-bounds indexing during capture.
        self.last_token_indices.zero_()
        self.idx_mapping.zero_()

        # Capture the prefill routine (model forward + compute_logits +
        # sample).
        # For FULL graphs, the entire routine is recorded as one graph.
        # For PIECEWISE, only the model's compiled regions are captured
        # and the rest (compute_logits, gumbel_sample) runs eagerly.
        # Draft prefill reuses the target model's attention metadata at
        # runtime, so capture builds its dummy metadata through the target
        # model runner's builders and buffers.
        assert self.prefill_cudagraph_manager is not None
        if self.prefill_cudagraph_manager.use_breakable_cg:
            self.prefill_cudagraph_manager.init_breakable_cg_runner(self.model)

        self.on_prefill_begin(self.max_num_reqs)
        self.prefill_cudagraph_manager.capture(
            self._prefill,
            self.model_state,
            self.target_input_buffers,
            self.block_tables,
            self.target_attn_groups,
            self.kv_cache_config,
            progress_bar_desc="Capturing prefill CUDA graphs",
        )
        self.on_prefill_end(self.max_num_reqs)

        if self.num_speculative_steps == 1:
            return

        self.on_multi_step_decode_begin(self.max_num_reqs)
        # Capture either the fused decode loop or one decode step per graph.
        assert self.decode_cudagraph_manager is not None
        decode_fn = (
            self._generate_fused_drafts
            if self.use_fused_multi_step_decode
            else self._generate_draft
        )
        self.decode_cudagraph_manager.capture(
            decode_fn,
            self.model_state,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            progress_bar_desc="Capturing decode CUDA graphs",
        )
        self.on_multi_step_decode_end(self.max_num_reqs)

    def dispatch_batch(
        self,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        *,
        decode: bool,
        need_eager: bool,
        dp_sync: DPSyncState | None,
    ) -> tuple[BatchExecutionDescriptor, DPSyncState | None]:
        """Decide the shape a draft pass runs at, agreed across DP ranks: the
        first pass over the target's tokens (`decode=False`) or one of the
        one-token-per-request passes after it (`decode=True`).

        Out-of-tree hardware speculators override this to pick from the
        shapes they compiled and to run their own DP agreement."""
        return dispatch_cg_and_sync_dp(
            self.decode_cudagraph_manager if decode else self.prefill_cudagraph_manager,
            num_reqs,
            num_tokens,
            uniform_token_count,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=need_eager,
            dp_sync=dp_sync,
        )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        # [num_tokens, hidden_size]
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size]
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        dp_sync: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        num_tokens = input_batch.num_tokens
        num_tokens_padded = input_batch.num_tokens_after_padding
        num_reqs = input_batch.num_reqs
        max_query_len = input_batch.num_scheduled_tokens.max()
        max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()
        self.draft_max_seq_len = min(
            max_seq_len + self.num_speculative_steps, self.max_model_len
        )

        # NOTE(woosuk): To avoid CPU-GPU synchronization without CPU knowing the
        # number of rejected tokens, we maintain the size of input_ids and
        # hidden_states the same as the target model's. This means, we pad each
        # request's query length to include any rejected positions. By doing so,
        # we can also reuse the attention metadata (e.g., query_start_loc,
        # seq_lens) of the target model.
        if aux_hidden_states:
            assert self.method == "eagle3"
            hidden_states = self.model.combine_hidden_states(
                torch.cat(aux_hidden_states, dim=-1)
            )
        else:
            hidden_states = last_hidden_states
        self._copy_request_inputs(
            num_reqs,
            input_batch.idx_mapping,
            temperature,
            seeds,
            dummy_run=dummy_run,
        )

        # Get the input ids and last token indices for the speculator.
        self.kernels.prepare_draft_prefill_inputs(
            self.last_token_indices,
            self.current_draft_step,
            self.input_buffers,
            input_batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            self.max_num_reqs,
        )

        if self.pcp_manager is not None:
            self.pcp_manager.prepare_draft_prefill(
                input_batch,
                self.input_buffers.input_ids[:num_tokens_padded],
            )
            prefill = self.pcp_manager.draft_prefill_batch
            if prefill is not None:
                num_tokens = prefill.num_tokens
                num_tokens_padded = prefill.num_tokens_after_padding
        self.hidden_states[:num_tokens_padded].copy_(hidden_states)

        # When all requests are decoding (no true prefills), each has
        # num_speculative_steps + 1 tokens, enabling FULL graph replay.
        uniform_token_count = get_uniform_decode_token_count(
            num_reqs,
            # Use the actual number of tokens without padding added by
            # the target model during FULL cudagraph.
            num_tokens,
            max_query_len,
            input_batch.has_prefill,
        )
        prefill_batch_desc, prefill_batch_sync = self.dispatch_batch(
            num_reqs,
            num_tokens_padded,
            uniform_token_count,
            decode=False,
            need_eager=is_profile,
            dp_sync=dp_sync,
        )
        num_tokens_across_dp = (
            prefill_batch_sync.num_tokens_across_dp
            if prefill_batch_sync is not None
            else None
        )

        self._prepare_eplb_forward(num_tokens)

        self.on_prefill_begin(num_reqs)
        if prefill_batch_desc.cg_mode == CUDAGraphMode.FULL:
            # Replay the full graph for draft prefill.
            assert self.prefill_cudagraph_manager is not None
            self.prefill_cudagraph_manager.run_fullgraph(prefill_batch_desc)
        else:
            # The target model's attention metadata and slot mappings
            # can directly be used for draft prefill, because of the
            # identical batch shape and KV cache layout.
            self._prefill(
                num_reqs,
                prefill_batch_desc.num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=prefill_batch_desc.cg_mode,
                mm_inputs=mm_inputs,
            )
        self.on_prefill_end(num_reqs)

        if self.num_speculative_steps == 1:
            # Early exit.
            return self.draft_tokens[:num_reqs, :1]

        if self.pcp_manager is not None and not dummy_run:
            self.block_tables.gather_block_tables(
                input_batch.idx_mapping, num_reqs_padded=num_reqs
            )

        # Prepare the inputs for the decode steps.
        self.kernels.prepare_draft_decode_inputs(
            self.draft_tokens[:num_reqs, 0],
            input_batch.seq_lens,
            num_rejected,
            self.input_buffers,
            self.sample_src_positions,
            self.max_model_len,
            self.max_num_reqs,
            advance_draft_positions=self.advance_draft_positions,
        )

        decode_batch_sync, num_batch_tokens = (
            self._build_uniform_batch_dp_sync(dp_sync, num_reqs, num_query_per_req=1)
            if dp_sync is not None
            else (None, num_reqs)
        )
        # Each request produces exactly 1 token per draft generation step,
        # enabling FULL graph replay.
        decode_batch_desc, decode_batch_sync = self.dispatch_batch(
            num_reqs,
            num_batch_tokens,
            1,
            decode=True,
            need_eager=is_profile,
            dp_sync=decode_batch_sync,
        )
        num_tokens_across_dp = (
            decode_batch_sync.num_tokens_across_dp
            if decode_batch_sync is not None
            else None
        )

        self.on_multi_step_decode_begin(num_reqs)
        # Generate the remaining num_speculative_steps - 1 draft tokens.
        decode_fn = (
            self._fused_multi_step_decode
            if self.use_fused_multi_step_decode
            else self._multi_step_decode
        )
        decode_fn(
            num_reqs,
            dummy_run and skip_attn_for_dummy_run,
            decode_batch_desc,
            num_tokens_across_dp,
            input_batch.seq_lens_cpu_upper_bound,
        )
        self.on_multi_step_decode_end(num_reqs)

        return self.draft_tokens[:num_reqs]

    @torch.inference_mode()
    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_buffers: InputBatch | InputBuffers = self.input_buffers
        is_padding = None
        if self.pcp_manager is not None:
            input_buffers = self.pcp_manager.get_draft_input_buffers(self.input_buffers)
            is_padding = input_buffers.is_padding[:num_tokens]
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
            is_padding=is_padding,
        ):
            inputs_embeds = None
            if self.supports_mm_inputs:
                assert self.inputs_embeds is not None
                # Merge multimodal embeddings with input ids.
                mm_embeds, is_mm_embed = mm_inputs or (None, None)
                num_input_tokens = (
                    is_mm_embed.shape[0] if is_mm_embed is not None else num_tokens
                )
                self.inputs_embeds[:num_input_tokens] = self.model.embed_input_ids(
                    input_buffers.input_ids[:num_input_tokens],
                    multimodal_embeddings=mm_embeds,
                    is_multimodal=is_mm_embed,
                )
                inputs_embeds = self.inputs_embeds[:num_tokens]

            model_inputs = dict(
                input_ids=input_buffers.input_ids[:num_tokens],
                positions=input_buffers.positions[:num_tokens],
                hidden_states=self.hidden_states[:num_tokens],
                inputs_embeds=inputs_embeds,
            )
            if cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
                # Draft prefill with PIECEWISE cudagraph (compiled PW or breakable),
                # chosen inside run_pw_graph.
                assert self.prefill_cudagraph_manager is not None
                ret_hidden_states = self.prefill_cudagraph_manager.run_pw_graph(
                    self.model, model_inputs
                )
            else:
                # Eager (NONE): call the raw model directly.
                ret_hidden_states = self.model(**model_inputs)
        # Some MTP models declare a single-tensor contract but return
        # (logits_hidden, feedback_hidden) for final-norm correctness.
        if isinstance(ret_hidden_states, tuple):
            last_hidden_states, hidden_states = ret_hidden_states
        else:
            last_hidden_states = ret_hidden_states
            hidden_states = ret_hidden_states
        return last_hidden_states, hidden_states

    def _prefill(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        last_token_indices = self.last_token_indices[:num_reqs]
        positions = self.input_buffers.positions[last_token_indices]
        # The output hidden state at position P (= positions) and the token id
        # at P+1 are used to draft the token at P+2. Sampling keys a draw by the
        # position before the sampled token, so the net adjustment is +1.
        sample_src_positions = positions + 1
        idx_mapping = self.idx_mapping[:num_reqs]

        last_hidden_states, hidden_states = self._run_model(
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            mm_inputs=mm_inputs,
        )
        if self.pcp_manager is not None:
            last_hidden_states, hidden_states = self.pcp_manager.restore_draft_prefill(
                last_hidden_states, hidden_states
            )

        sample_hidden_states = last_hidden_states[last_token_indices]
        self.draft_tokens[:num_reqs, 0] = self.sample_draft(
            sample_hidden_states,
            sample_src_positions,
            idx_mapping,
            self.temperature,
            self.seeds,
            self.current_draft_step,
            self.draft_logits,
        )
        if last_hidden_states is hidden_states:
            self.hidden_states[:num_reqs] = sample_hidden_states
        else:
            self.hidden_states[:num_reqs] = hidden_states[last_token_indices]
        self.input_buffers.positions[:num_reqs] = positions
        self.sample_src_positions[:num_reqs] = sample_src_positions

    def _multi_step_decode(
        self,
        num_reqs: int,
        skip_attn: bool,
        batch_desc: BatchExecutionDescriptor,
        num_tokens_across_dp: torch.Tensor | None,
        seq_lens_cpu_upper_bound: torch.Tensor,
    ) -> None:
        positions = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        idx_mapping = self.idx_mapping[:num_reqs]

        attn_metadata = None
        slot_mappings_by_layer = None
        for step in range(1, self.num_speculative_steps):
            # Rebuild every step when positions advance, or just once
            # on the first step when positions are constant (Gemma4 MTP).
            if not skip_attn and (self.advance_draft_positions or step == 1):
                slot_mappings = self.block_tables.compute_slot_mappings(
                    idx_mapping,
                    query_start_loc,
                    positions,
                    batch_desc.num_tokens,
                )
                slot_mappings_by_layer = build_slot_mappings_by_layer(
                    slot_mappings, self.kv_cache_config
                )
                attn_metadata = self._build_uniform_attn_metadata(
                    num_reqs=num_reqs,
                    batch_desc=batch_desc,
                    num_query_per_req=1,
                    seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                    step=step,
                )

            self.current_draft_step.fill_(step)

            if batch_desc.cg_mode == CUDAGraphMode.FULL:
                assert self.decode_cudagraph_manager is not None
                self.decode_cudagraph_manager.run_fullgraph(batch_desc)
            else:
                self._generate_draft(
                    num_reqs,
                    batch_desc.num_tokens,
                    attn_metadata,
                    slot_mappings_by_layer,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=batch_desc.cg_mode,
                )

    def _fused_multi_step_decode(
        self,
        num_reqs: int,
        skip_attn: bool,
        batch_desc: BatchExecutionDescriptor,
        num_tokens_across_dp: torch.Tensor | None,
        seq_lens_cpu_upper_bound: torch.Tensor,
    ) -> None:
        positions = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        idx_mapping = self.idx_mapping[:num_reqs]

        attn_metadata = None
        slot_mappings_by_layer = None
        if not skip_attn:
            slot_mappings = self.block_tables.compute_slot_mappings(
                idx_mapping,
                query_start_loc,
                positions,
                batch_desc.num_tokens,
            )
            if batch_desc.cg_mode != CUDAGraphMode.FULL:
                slot_mappings_by_layer = build_slot_mappings_by_layer(
                    slot_mappings, self.kv_cache_config
                )
            attn_metadata = self._build_uniform_attn_metadata(
                num_reqs=num_reqs,
                batch_desc=batch_desc,
                num_query_per_req=1,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=1,
            )

        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            assert self.decode_cudagraph_manager is not None
            self.decode_cudagraph_manager.run_fullgraph(batch_desc)
            return

        self._generate_fused_drafts(
            num_reqs,
            batch_desc.num_tokens,
            attn_metadata,
            slot_mappings_by_layer,
            num_tokens_across_dp,
            batch_desc.cg_mode,
        )

    def _generate_fused_drafts(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        idx_mapping = self.idx_mapping[:num_reqs]
        positions = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        attn_groups = (
            [group for groups in self.attn_groups for group in groups]
            if attn_metadata is not None
            else []
        )

        for step in range(1, self.num_speculative_steps):
            self.current_draft_step.fill_(step)
            self._generate_draft(
                num_reqs,
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
            if (
                step < self.num_speculative_steps - 1
                and attn_metadata is not None
                and self.advance_draft_positions
            ):
                self.block_tables.compute_slot_mappings(
                    idx_mapping,
                    query_start_loc,
                    positions,
                    num_tokens_padded,
                )
                for attn_group in attn_groups:
                    attn_group.update_draft_decode_metadata(attn_metadata)

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        self._prepare_eplb_forward(num_reqs)

        idx_mapping = self.idx_mapping[:num_reqs]
        # Run the draft model forward pass.
        last_hidden_states, hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )

        # Sample the draft tokens.
        sample_hidden_states = last_hidden_states[:num_reqs]
        sample_src_positions = self.sample_src_positions[:num_reqs]
        draft_tokens = self.sample_draft(
            sample_hidden_states,
            sample_src_positions,
            idx_mapping,
            self.temperature,
            self.seeds,
            self.current_draft_step,
            self.draft_logits,
        )

        # Update the inputs for the next step.
        self.kernels.update_draft_inputs(
            draft_tokens,
            self.current_draft_step,
            hidden_states,
            self.draft_tokens,
            self.hidden_states,
            self.input_buffers,
            self.sample_src_positions,
            num_reqs,
            self.max_model_len,
            self.num_speculative_steps,
            advance_draft_positions=self.advance_draft_positions,
        )

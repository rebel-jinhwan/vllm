# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.config import SpeculativeConfig
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    rejection_sample,
)


@triton.jit
def _flatten_sampled_kernel(
    # [num_logits]
    flat_sampled_ptr,
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    for i in range(num_sampled):
        token_id = tl.load(sampled_ptr + req_idx * sampled_stride + i)
        tl.store(flat_sampled_ptr + start_idx + i, token_id)


class RejectionSampler:
    def __init__(
        self,
        sampler: Sampler,
        spec_config: SpeculativeConfig,
        device: torch.device,
    ):
        self.sampler = sampler
        self.num_speculative_steps = spec_config.num_speculative_tokens
        rejection_sample_method = spec_config.rejection_sample_method
        self.use_block_verification: bool = False
        self.synthetic_conditional_rates: torch.Tensor | None = None
        if rejection_sample_method == "synthetic":
            assert spec_config.synthetic_acceptance_rates is not None
            self.synthetic_conditional_rates = torch.tensor(
                unconditional_to_conditional_rates(
                    spec_config.synthetic_acceptance_rates
                ),
                dtype=torch.float32,
                device=device,
            )
        elif rejection_sample_method == "block":
            self.use_block_verification = True

    def _get_logprobs_tensors(
        self,
        input_batch: InputBatch,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        logits: torch.Tensor,
    ) -> LogprobsTensors | None:
        max_num_logprobs = self.sampler.sampling_states.max_num_logprobs(
            input_batch.idx_mapping_np
        )
        if max_num_logprobs == NO_LOGPROBS:
            return None

        num_logits = logits.shape[0]
        flat_sampled = torch.zeros(
            num_logits, dtype=sampled.dtype, device=sampled.device
        )
        self.flatten_sampled(
            flat_sampled, sampled, num_sampled, input_batch.cu_num_logits
        )
        expanded_logits = num_logits != input_batch.idx_mapping.shape[0]
        return self.sampler.compute_topk_scores(
            logits,
            max_num_logprobs,
            flat_sampled,
            input_batch.cu_num_logits_np.tolist() if expanded_logits else None,
            logits_mode=self.sampler.logprobs_mode
            in ("raw_logits", "processed_logits"),
        )

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None = None,
    ) -> SamplerOutput:
        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = (
            self.sampler.get_num_nans(logits) if self.sampler.compute_nans else None
        )

        draft_sampled = input_batch.input_ids[input_batch.logits_indices]
        pos = input_batch.positions[input_batch.logits_indices]
        processed_logits = self.sampler.apply_sampling_params(
            logits,
            input_batch.expanded_idx_mapping,
            input_batch.idx_mapping_np,
            pos,
            draft_sampled,
            input_batch.expanded_local_pos,
        )
        sampled, num_sampled = self.rejection_sample(
            processed_logits,
            draft_logits,
            draft_sampled,
            input_batch.cu_num_logits,
            pos,
            input_batch.idx_mapping,
            input_batch.expanded_idx_mapping,
            input_batch.expanded_local_pos,
            self.sampler.sampling_states.temperature.gpu,
            self.sampler.sampling_states.seeds.gpu,
            self.num_speculative_steps,
            self.synthetic_conditional_rates,
            use_fp64=self.sampler.use_fp64_gumbel,
            use_block_verification=self.use_block_verification,
        )
        logprobs_tensors = self._get_logprobs_tensors(
            input_batch,
            sampled,
            num_sampled,
            processed_logits
            if self.sampler.logprobs_mode in ("processed_logprobs", "processed_logits")
            else logits,
        )

        num_sampled, num_rejected = self.sampler.get_num_sampled_and_rejected(
            num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.sampler.req_states.prefill_len.gpu,
        )

        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    # --- Kernels. A platform without Triton subclasses the rejection sampler
    # and implements these. ---

    def rejection_sample(
        self,
        target_logits: torch.Tensor,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        cu_num_logits: torch.Tensor,
        pos: torch.Tensor,
        idx_mapping: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        temperature: torch.Tensor,
        seed: torch.Tensor,
        num_speculative_steps: int,
        synthetic_conditional_rates: torch.Tensor | None = None,
        use_fp64: bool = False,
        use_block_verification: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return rejection_sample(
            target_logits,
            draft_logits,
            draft_sampled,
            cu_num_logits,
            pos,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            temperature,
            seed,
            num_speculative_steps,
            synthetic_conditional_rates,
            use_fp64=use_fp64,
            use_block_verification=use_block_verification,
        )

    def flatten_sampled(
        self,
        flat_sampled: torch.Tensor,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        cu_num_logits: torch.Tensor,
    ) -> None:
        num_reqs = num_sampled.shape[0]
        _flatten_sampled_kernel[(num_reqs,)](
            flat_sampled,
            sampled,
            sampled.stride(0),
            num_sampled,
            cu_num_logits,
            num_warps=1,
        )

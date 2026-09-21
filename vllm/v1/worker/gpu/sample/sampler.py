# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

import vllm.envs as envs
from vllm.config.model import LogprobsMode
from vllm.sampling_params import SamplingParams
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.ops.topk_topp_sampler import (
    apply_top_k_top_p,
    flashinfer_sample,
    flashinfer_sampler_supported,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, get_num_sampled_and_rejected
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState, apply_bad_words
from vllm.v1.worker.gpu.sample.gumbel import apply_temperature, gumbel_sample
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState, apply_logit_bias
from vllm.v1.worker.gpu.sample.logprob import (
    LogprobTokenIdsState,
    compute_token_logprobs,
    compute_token_ranks,
    fill_logprob_token_ids,
)
from vllm.v1.worker.gpu.sample.min_p import apply_min_p
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.penalties import (
    PenaltiesState,
    apply_penalties,
    bincount,
)
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS, SamplingStates
from vllm.v1.worker.gpu.states import RequestState


class Sampler:
    def __init__(
        self,
        max_num_reqs: int,
        vocab_size: int,
        device: torch.device,
        req_states: RequestState,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        num_speculative_tokens: int = 1,
        use_fp64_gumbel: bool = False,
    ):
        self.logprobs_mode = logprobs_mode
        self.compute_nans = envs.VLLM_COMPUTE_NANS_IN_LOGITS  # False by default.
        self.use_fp64_gumbel = use_fp64_gumbel

        self.req_states = req_states
        self.sampling_states = SamplingStates(max_num_reqs, vocab_size, self)
        self.penalties_state = PenaltiesState(req_states, self)
        self.logit_bias_state = LogitBiasState(max_num_reqs, device, self)
        self.bad_words_state = BadWordsState(req_states, self)
        self.logprob_token_ids_state = LogprobTokenIdsState(max_num_reqs, device)
        self.num_speculative_tokens = num_speculative_tokens
        self.use_flashinfer = flashinfer_sampler_supported()

    def add_request(
        self, req_idx: int, prompt_len: int, sampling_params: SamplingParams
    ) -> None:
        self.sampling_states.add_request(req_idx, sampling_params)
        self.penalties_state.add_request(req_idx, sampling_params)
        self.logit_bias_state.add_request(req_idx, prompt_len, sampling_params)
        self.bad_words_state.add_request(req_idx, sampling_params)
        self.logprob_token_ids_state.add_request(req_idx, sampling_params)

    def apply_staged_writes(self) -> None:
        self.sampling_states.apply_staged_writes()
        self.penalties_state.apply_staged_writes()
        self.logit_bias_state.apply_staged_writes()
        self.bad_words_state.apply_staged_writes()
        self.logprob_token_ids_state.apply_staged_writes()

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
    ) -> SamplerOutput:
        expanded_idx_mapping = input_batch.expanded_idx_mapping
        idx_mapping_np = input_batch.idx_mapping_np
        cu_num_logits_np = input_batch.cu_num_logits_np
        expanded_local_pos = input_batch.expanded_local_pos
        pos = input_batch.positions[input_batch.logits_indices]
        input_ids = input_batch.input_ids[input_batch.logits_indices]

        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = self.get_num_nans(logits) if self.compute_nans else None

        return_logprobs = self.returns_logprobs(idx_mapping_np)

        sampled, processed_logits = self.sample(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
            return_logprobs=return_logprobs,
        )

        if return_logprobs:
            if self.logprobs_mode in ("processed_logprobs", "processed_logits"):
                logits = processed_logits
            max_num_logprobs = self.sampling_states.max_num_logprobs(idx_mapping_np)
            max_per_req_token_ids = self.logprob_token_ids_state.max_num_token_ids(
                idx_mapping_np
            )
            expanded_logits = logits.shape[0] != idx_mapping_np.shape[0]
            cu_num_logits = cu_num_logits_np.tolist() if expanded_logits else None
            num_logprobs = max_num_logprobs if max_num_logprobs != NO_LOGPROBS else 0
            logprobs_tensors = self.compute_topk_scores(
                logits,
                num_logprobs,
                sampled,
                cu_num_logits,
                logprob_token_ids_state=self.logprob_token_ids_state,
                expanded_idx_mapping=input_batch.expanded_idx_mapping,
                max_per_req_token_ids=max_per_req_token_ids,
                logits_mode=self.logprobs_mode in ("raw_logits", "processed_logits"),
            )
        else:
            logprobs_tensors = None

        # 1 sampled token per request, except chunked-prefill requests
        # (seq_len < prefill_len) which aren't done prefilling and produce no
        # output token. num_rejected is always 0 here (one logit per request).
        num_sampled, num_rejected = self.get_num_sampled_and_rejected(
            input_batch.seq_lens.new_ones(input_batch.num_reqs),
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.req_states.prefill_len.gpu,
        )

        # These are GPU tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.view(-1, 1),
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )
        return sampler_output

    def returns_logprobs(self, idx_mapping_np: np.ndarray) -> bool:
        """Whether any request in the batch produces logprobs this step."""
        return (
            self.sampling_states.max_num_logprobs(idx_mapping_np) != NO_LOGPROBS
            or self.logprob_token_ids_state.max_num_token_ids(idx_mapping_np) > 0
        )

    def apply_sampling_params(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        skip_top_k_top_p: bool = False,
    ) -> torch.Tensor:
        # The ops below upcast to fp32 internally, so the input dtype is kept and
        # mutated in place. Only raw_logprobs reads the unmodified logits
        # afterward, so copy just for that case.
        if self.logprobs_mode.startswith("raw_") and self.returns_logprobs(
            idx_mapping_np
        ):
            logits = logits.clone()

        # Apply logit bias (e.g., allowed_token_ids, min_tokens) in place.
        self.logit_bias_state.apply_logit_bias(
            logits, expanded_idx_mapping, idx_mapping_np, pos
        )

        # Apply penalties in place.
        self.penalties_state.apply_penalties(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            input_ids,
            expanded_local_pos,
        )

        # Apply bad words masking in place.
        self.bad_words_state.apply_bad_words(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            input_ids,
            expanded_local_pos,
        )

        # Apply temperature in place.
        self.sampling_states.apply_temperature(
            logits, expanded_idx_mapping, idx_mapping_np
        )

        # Apply min_p in place.
        self.sampling_states.apply_min_p(logits, expanded_idx_mapping, idx_mapping_np)

        if skip_top_k_top_p:
            return logits

        # Apply top_k and/or top_p. This might or might not return a new tensor.
        return self.sampling_states.apply_top_k_top_p(
            logits, expanded_idx_mapping, idx_mapping_np
        )

    def sample(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        return_logprobs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        processed_logits = self.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
            skip_top_k_top_p=True,
        )
        top_k, top_p = self.sampling_states.get_top_k_top_p(
            expanded_idx_mapping, idx_mapping_np
        )
        use_flashinfer = self.use_flashinfer and not (
            # Don't use FI sampler if no requests use top_k/top_p, if there are
            # any greedy requests or per-request seeds, or if post-processed
            # logprobs need to be returned for any requests.
            (top_k is None and top_p is None)
            or (
                return_logprobs
                and self.logprobs_mode in ("processed_logprobs", "processed_logits")
            )
            or self.sampling_states.any_greedy(idx_mapping_np)
            or self.sampling_states.any_explicit_seed(idx_mapping_np)
        )

        # Sample the next token.
        if use_flashinfer:
            sampled = flashinfer_sample(processed_logits, top_k, top_p).to(torch.int64)
        else:
            processed_logits = apply_top_k_top_p(processed_logits, top_k, top_p)
            sampled = self.gumbel_sample(
                processed_logits,
                expanded_idx_mapping,
                self.sampling_states.temperature.gpu,
                self.sampling_states.seeds.gpu,
                pos,
                apply_temperature=False,
                use_fp64=self.use_fp64_gumbel,
            )
        return sampled, processed_logits

    # --- Kernels. A platform without Triton subclasses the sampler and
    # implements these; everything above is device-neutral. ---

    def apply_temperature(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
    ) -> None:
        apply_temperature(logits, expanded_idx_mapping, temperature)

    def apply_min_p(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        min_p: torch.Tensor,
    ) -> None:
        apply_min_p(logits, expanded_idx_mapping, min_p)

    def apply_penalties(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        token_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        repetition_penalty: torch.Tensor,
        frequency_penalty: torch.Tensor,
        presence_penalty: torch.Tensor,
        prompt_bin_mask: torch.Tensor,
        output_bin_counts: torch.Tensor,
    ) -> None:
        apply_penalties(
            logits,
            expanded_idx_mapping,
            token_ids,
            expanded_local_pos,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            prompt_bin_mask,
            output_bin_counts,
        )

    def bincount(
        self,
        expanded_idx_mapping: torch.Tensor,
        all_token_ids: torch.Tensor,
        prompt_len: torch.Tensor,
        prefill_len: torch.Tensor,
        prompt_bin_mask: torch.Tensor,
        output_bin_counts: torch.Tensor,
        max_prefill_len: int,
    ) -> None:
        bincount(
            expanded_idx_mapping,
            all_token_ids,
            prompt_len,
            prefill_len,
            prompt_bin_mask,
            output_bin_counts,
            max_prefill_len,
        )

    def apply_logit_bias(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        pos: torch.Tensor,
        num_allowed_token_ids: torch.Tensor,
        allowed_token_ids: torch.Tensor,
        num_logit_bias: torch.Tensor,
        logit_bias_token_ids: torch.Tensor,
        logit_bias: torch.Tensor,
        min_lens: torch.Tensor,
        num_stop_token_ids: torch.Tensor,
        stop_token_ids: torch.Tensor,
    ) -> None:
        apply_logit_bias(
            logits,
            expanded_idx_mapping,
            pos,
            num_allowed_token_ids,
            allowed_token_ids,
            num_logit_bias,
            logit_bias_token_ids,
            logit_bias,
            min_lens,
            num_stop_token_ids,
            stop_token_ids,
        )

    def apply_bad_words(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        bad_word_token_ids: torch.Tensor,
        bad_word_offsets: torch.Tensor,
        num_bad_words: torch.Tensor,
        all_token_ids: torch.Tensor,
        prompt_len: torch.Tensor,
        total_len: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        max_num_bad_words: int,
    ) -> None:
        apply_bad_words(
            logits,
            expanded_idx_mapping,
            bad_word_token_ids,
            bad_word_offsets,
            num_bad_words,
            all_token_ids,
            prompt_len,
            total_len,
            input_ids,
            expanded_local_pos,
            max_num_bad_words,
        )

    def gumbel_sample(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seed: torch.Tensor,
        pos: torch.Tensor,
        apply_temperature: bool,
        output_processed_logits: torch.Tensor | None = None,
        output_processed_logits_col: torch.Tensor | None = None,
        use_fp64: bool = False,
    ) -> torch.Tensor:
        return gumbel_sample(
            logits,
            expanded_idx_mapping,
            temperature,
            seed,
            pos,
            apply_temperature,
            output_processed_logits,
            output_processed_logits_col,
            use_fp64,
        )

    def compute_token_logprobs(
        self, logits: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        return compute_token_logprobs(logits, token_ids)

    def compute_token_ranks(
        self, logits: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        return compute_token_ranks(logits, token_ids)

    def fill_logprob_token_ids(
        self,
        out_token_ids: torch.Tensor,
        out_valid_mask: torch.Tensor,
        sampled_token_ids: torch.Tensor,
        topk_token_ids: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        num_per_req_token_ids: torch.Tensor,
        per_req_token_ids: torch.Tensor,
        num_topk: int,
    ) -> None:
        fill_logprob_token_ids(
            out_token_ids,
            out_valid_mask,
            sampled_token_ids,
            topk_token_ids,
            expanded_idx_mapping,
            num_per_req_token_ids,
            per_req_token_ids,
            num_topk,
        )

    def get_num_nans(self, logits: torch.Tensor) -> torch.Tensor:
        return get_num_nans(logits)

    def get_num_sampled_and_rejected(
        self,
        num_sampled: torch.Tensor,
        seq_lens: torch.Tensor,
        cu_num_logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        prefill_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return get_num_sampled_and_rejected(
            num_sampled, seq_lens, cu_num_logits, idx_mapping, prefill_len
        )

    def compute_topk_scores(
        self,
        logits: torch.Tensor,
        num_logprobs: int,
        sampled_token_ids: torch.Tensor,
        cu_num_logits: list[int] | None = None,
        logprob_token_ids_state: LogprobTokenIdsState | None = None,
        expanded_idx_mapping: torch.Tensor | None = None,
        max_per_req_token_ids: int = 0,
        logits_mode: bool = False,
    ) -> LogprobsTensors:
        assert num_logprobs >= 0
        batch_size = logits.shape[0]
        if max_per_req_token_ids == 0:
            # Fast path: no request asked for custom logprob_token_ids.
            logprob_token_ids = sampled_token_ids.unsqueeze(-1)
            if num_logprobs > 0:
                topk_indices = torch.topk(logits, num_logprobs, dim=-1).indices
                logprob_token_ids = torch.cat((logprob_token_ids, topk_indices), dim=1)
            if logits_mode:
                scores = logits.gather(-1, logprob_token_ids).to(torch.float32)
            else:
                scores = self.compute_token_logprobs(logits, logprob_token_ids)
        else:
            # Some requests specified logprob_token_ids: build the [batch_size,
            # 1 + max_cols] token_ids matrix and validity mask, overriding the
            # topk columns with per-request tokens where applicable.
            assert logprob_token_ids_state is not None
            assert expanded_idx_mapping is not None
            if num_logprobs > 0:
                topk_token_ids = torch.topk(logits, num_logprobs, dim=-1).indices
                topk_token_ids = topk_token_ids.to(torch.int32)
            else:
                topk_token_ids = logprob_token_ids_state.token_ids.gpu
            num_cols = max(num_logprobs, max_per_req_token_ids)
            logprob_token_ids = sampled_token_ids.new_zeros((batch_size, 1 + num_cols))
            valid_mask = torch.zeros_like(logprob_token_ids, dtype=torch.bool)
            self.fill_logprob_token_ids(
                logprob_token_ids,
                valid_mask,
                sampled_token_ids,
                topk_token_ids,
                expanded_idx_mapping,
                logprob_token_ids_state.num_token_ids.gpu,
                logprob_token_ids_state.token_ids.gpu,
                num_logprobs,
            )
            if logits_mode:
                scores = logits.gather(-1, logprob_token_ids).to(torch.float32)
            else:
                scores = self.compute_token_logprobs(logits, logprob_token_ids)
            scores = scores.masked_fill(~valid_mask, float("-inf"))
        return LogprobsTensors(
            logprob_token_ids=logprob_token_ids,
            logprobs=scores,
            selected_token_ranks=self.compute_token_ranks(logits, sampled_token_ids),
            cu_num_generated_tokens=cu_num_logits,
        )

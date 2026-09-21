# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The device kernels of the V2 model runner (vllm/v1/worker/gpu/).

The runner, sampler, rejection sampler, prompt-logprobs and structured-outputs
workers and the speculators reach every kernel through this interface. A
platform without Triton returns its own implementation from
`GPUModelRunner.init_kernels()`; everything else in the runner stays as it is.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers


class ModelRunnerKernels(ABC):
    # --- Request state and input preparation (GPUModelRunner). ---

    @abstractmethod
    def prepare_prefill_inputs(
        self,
        input_ids: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        all_token_ids: torch.Tensor,
        prefill_len: torch.Tensor,
        num_computed_tokens: torch.Tensor,
    ) -> None: ...

    @abstractmethod
    def prepare_pos_seq_lens(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        pos: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None: ...

    @abstractmethod
    def combine_sampled_and_draft_tokens(
        self,
        input_ids: torch.Tensor,
        idx_mapping: torch.Tensor,
        last_sampled_tokens: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        prefill_len: torch.Tensor,
        draft_tokens: torch.Tensor,
        cu_num_logits: torch.Tensor,
        num_logits: int,
        num_new_sampled_tokens: int,
    ) -> torch.Tensor: ...

    @abstractmethod
    def expand_idx_mapping(
        self,
        idx_mapping: torch.Tensor,
        total_num_logits: int,
        cu_num_logits: torch.Tensor,
        max_expand_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def post_update(
        self,
        idx_mapping: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        last_sampled_tokens: torch.Tensor,
        output_bin_counts: torch.Tensor | None,
        sampled_tokens: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        query_start_loc: torch.Tensor | None,
        all_token_ids: torch.Tensor,
        total_len: torch.Tensor,
    ) -> None: ...

    @abstractmethod
    def post_update_num_computed_tokens(
        self,
        idx_mapping: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> None: ...

    # --- Sampling (Sampler and its states). ---

    @abstractmethod
    def apply_temperature(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
    ) -> None: ...

    @abstractmethod
    def apply_min_p(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        min_p: torch.Tensor,
    ) -> None: ...

    @abstractmethod
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
    ) -> None: ...

    @abstractmethod
    def bincount(
        self,
        expanded_idx_mapping: torch.Tensor,
        all_token_ids: torch.Tensor,
        prompt_len: torch.Tensor,
        prefill_len: torch.Tensor,
        prompt_bin_mask: torch.Tensor,
        output_bin_counts: torch.Tensor,
        max_prefill_len: int,
    ) -> None: ...

    @abstractmethod
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
    ) -> None: ...

    @abstractmethod
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
    ) -> None: ...

    @abstractmethod
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
    ) -> torch.Tensor: ...

    @abstractmethod
    def compute_token_logprobs(
        self, logits: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor: ...

    @abstractmethod
    def compute_token_ranks(
        self, logits: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor: ...

    @abstractmethod
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
    ) -> None: ...

    @abstractmethod
    def get_num_nans(self, logits: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def get_num_sampled_and_rejected(
        self,
        num_sampled: torch.Tensor,
        seq_lens: torch.Tensor,
        cu_num_logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        prefill_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    # --- Prompt logprobs (PromptLogprobsWorker). ---

    @abstractmethod
    def get_prompt_logprobs_token_ids(
        self,
        num_tokens: int,
        query_start_loc: torch.Tensor,
        idx_mapping: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        all_token_ids: torch.Tensor,
    ) -> torch.Tensor: ...

    # --- Rejection sampling (RejectionSampler). ---

    @abstractmethod
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
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def flatten_sampled(
        self,
        flat_sampled: torch.Tensor,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        cu_num_logits: torch.Tensor,
    ) -> None: ...

    # --- Structured outputs (StructuredOutputsWorker). ---

    @abstractmethod
    def apply_grammar_bitmask(
        self, logits: torch.Tensor, logits_indices: torch.Tensor, bitmask: torch.Tensor
    ) -> None: ...

    # --- Draft input preparation (AutoRegressiveSpeculator). ---

    @abstractmethod
    def prepare_draft_prefill_inputs(
        self,
        last_token_indices: torch.Tensor,
        current_draft_step: torch.Tensor,
        input_buffers: "InputBuffers",
        input_batch: "InputBatch",
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        max_num_reqs: int,
    ) -> torch.Tensor: ...

    @abstractmethod
    def prepare_draft_decode_inputs(
        self,
        draft_tokens: torch.Tensor,
        target_seq_lens: torch.Tensor,
        num_rejected: torch.Tensor,
        input_buffers: "InputBuffers",
        max_model_len: int,
        max_num_reqs: int,
        advance_draft_positions: bool = True,
    ) -> None: ...

    @abstractmethod
    def update_draft_inputs(
        self,
        draft_tokens: torch.Tensor,
        current_draft_step: torch.Tensor,
        hidden_states: torch.Tensor,
        output_draft_tokens: torch.Tensor,
        next_input_hidden_states: torch.Tensor,
        input_buffers: "InputBuffers",
        num_reqs: int,
        max_model_len: int,
        num_speculative_steps: int,
        advance_draft_positions: bool = True,
    ) -> None: ...

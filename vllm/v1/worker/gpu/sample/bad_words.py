# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor
from vllm.v1.worker.gpu.sample.logits_processor.interface import (
    LogitsContext,
    LogitsProcessor,
    LogitsProcRequestState,
)
from vllm.v1.worker.kernels import ModelRunnerKernels

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class BadWordsState(LogitsProcessor):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        req_states: LogitsProcRequestState,
        kernels: ModelRunnerKernels,
    ):
        self.kernels = kernels
        self.req_states = req_states
        max_num_reqs = req_states.max_num_reqs
        device = req_states.device

        max_total_tokens = envs.VLLM_MAX_BAD_WORDS_TOTAL_TOKENS
        max_num_bad_words = envs.VLLM_MAX_NUM_BAD_WORDS
        # flattened bad word tokens: [max_num_reqs, VLLM_MAX_BAD_WORDS_TOTAL_TOKENS]
        self.bad_word_token_ids = StagedWriteTensor(
            (max_num_reqs, max_total_tokens), dtype=torch.int32, device=device
        )
        # cumulative offsets of bad words: [max_num_reqs, VLLM_MAX_NUM_BAD_WORDS + 1]
        self.bad_word_offsets = StagedWriteTensor(
            (max_num_reqs, max_num_bad_words + 1), dtype=torch.int32, device=device
        )
        # number of bad words per request
        self.num_bad_words = UvaBackedTensor(max_num_reqs, dtype=torch.int32)

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        bad_words_token_ids = sampling_params.bad_words_token_ids
        if not bad_words_token_ids:
            self.num_bad_words.np[req_idx] = 0
            return False

        num_bad_words = len(bad_words_token_ids)
        max_num_bad_words = envs.VLLM_MAX_NUM_BAD_WORDS
        if num_bad_words > max_num_bad_words:
            raise ValueError(
                f"Too many bad words: {num_bad_words}. "
                f"The max number is {max_num_bad_words}."
            )

        # Flatten bad words and compute offsets
        flattened_tokens: list[int] = []
        offsets: list[int] = [0]
        for bad_word in bad_words_token_ids:
            flattened_tokens.extend(bad_word)
            offsets.append(len(flattened_tokens))

        max_total_tokens = envs.VLLM_MAX_BAD_WORDS_TOTAL_TOKENS
        if len(flattened_tokens) > max_total_tokens:
            raise ValueError(
                f"Too many total bad word tokens: {len(flattened_tokens)}. "
                f"The max is {max_total_tokens}."
            )

        # Stage writes
        self.bad_word_token_ids.stage_write(req_idx, 0, flattened_tokens)
        self.bad_word_offsets.stage_write(req_idx, 0, offsets)
        self.num_bad_words.np[req_idx] = num_bad_words
        return True

    def apply_staged_writes(self) -> None:
        self.num_bad_words.copy_to_uva()
        self.bad_word_token_ids.apply_write()
        self.bad_word_offsets.apply_write()

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        max_num_bad_words = int(self.num_bad_words.np[ctx.idx_mapping_np].max())
        if max_num_bad_words == 0:
            # No request uses bad words. Skip the kernel launch.
            return logits

        self.kernels.apply_bad_words(
            logits,
            ctx.expanded_idx_mapping,
            self.bad_word_token_ids.gpu,
            self.bad_word_offsets.gpu,
            self.num_bad_words.gpu,
            self.req_states.all_token_ids.gpu,
            self.req_states.prompt_len.gpu,
            self.req_states.total_len.gpu,
            ctx.input_ids,
            ctx.expanded_local_pos,
            max_num_bad_words,
        )
        return logits

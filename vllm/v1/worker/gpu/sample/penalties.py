# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.sample.logits_processor.interface import (
    LogitsContext,
    LogitsProcessor,
    LogitsProcRequestState,
)
from vllm.v1.worker.kernels import ModelRunnerKernels

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class PenaltiesState(LogitsProcessor):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        req_states: LogitsProcRequestState,
        kernels: ModelRunnerKernels,
    ):
        self.kernels = kernels
        self.req_states = req_states
        self.device = req_states.device
        max_num_reqs = req_states.max_num_reqs
        vocab_size = req_states.vocab_size

        self.repetition_penalty = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.frequency_penalty = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.presence_penalty = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.use_penalty = np.zeros(max_num_reqs, dtype=bool)

        # Initialize repetition penalty manually because 0 is an invalid value for it.
        self.repetition_penalty.np.fill(1.0)
        self.repetition_penalty.copy_to_uva()

        # Statistics for penalties.
        self.prompt_bin_mask = torch.zeros(
            max_num_reqs, cdiv(vocab_size, 32), dtype=torch.int32, device=self.device
        )
        # TODO(woosuk): This tensor is rarely used but can be very large, taking up
        # GBs of GPU memory. Optimize the memory usage.
        self.output_bin_counts = torch.zeros(
            max_num_reqs, vocab_size, dtype=torch.int32, device=self.device
        )

        self._new_penalties_reqs: list[int] = []

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        self.repetition_penalty.np[req_idx] = sampling_params.repetition_penalty
        self.frequency_penalty.np[req_idx] = sampling_params.frequency_penalty
        self.presence_penalty.np[req_idx] = sampling_params.presence_penalty

        do_penalty = use_penalty(sampling_params)
        self.use_penalty[req_idx] = do_penalty
        if do_penalty:
            self._new_penalties_reqs.append(req_idx)
        return do_penalty

    def apply_staged_writes(self) -> None:
        if self._new_penalties_reqs:
            idx_mapping = async_tensor_h2d(
                self._new_penalties_reqs, dtype=torch.int32, device=self.device
            )

            prefill_lens = self.req_states.prefill_len.np[self._new_penalties_reqs]
            max_prefill_len = int(prefill_lens.max())
            self.kernels.bincount(
                idx_mapping,
                self.req_states.all_token_ids.gpu,
                self.req_states.prompt_len.gpu,
                self.req_states.prefill_len.gpu,
                self.prompt_bin_mask,
                self.output_bin_counts,
                max_prefill_len,
            )
            self._new_penalties_reqs.clear()

        self.repetition_penalty.copy_to_uva()
        self.frequency_penalty.copy_to_uva()
        self.presence_penalty.copy_to_uva()

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        if not np.any(self.use_penalty[ctx.idx_mapping_np]):
            # No request uses penalties. Skip the kernel launch.
            return logits

        self.kernels.apply_penalties(
            logits,
            ctx.expanded_idx_mapping,
            ctx.input_ids,
            ctx.expanded_local_pos,
            self.repetition_penalty.gpu,
            self.frequency_penalty.gpu,
            self.presence_penalty.gpu,
            self.prompt_bin_mask,
            self.output_bin_counts,
        )
        return logits


def use_penalty(sampling_params: SamplingParams) -> bool:
    return (
        sampling_params.repetition_penalty != 1.0
        or sampling_params.frequency_penalty != 0.0
        or sampling_params.presence_penalty != 0.0
    )

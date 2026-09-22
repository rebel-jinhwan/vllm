# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import PIN_MEMORY, async_tensor_h2d
from vllm.v1.worker.gpu.async_utils import stream
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.kernels import ModelRunnerKernels


def _build_grammar_mapping(
    req_ids: list[str],
    grammar_req_ids: list[str],
    cu_num_logits_np: np.ndarray,
    num_draft_tokens_per_req: np.ndarray | None,
    num_bonus_tokens: int,
    mask_stride: int,
) -> list[int]:
    mapping: list[int] = []
    req_id_to_idx = {req_id: i for i, req_id in enumerate(req_ids)}
    for grammar_req_id in grammar_req_ids:
        req_idx = req_id_to_idx[grammar_req_id]
        if num_draft_tokens_per_req is None:
            num_positions = int(
                cu_num_logits_np[req_idx + 1] - cu_num_logits_np[req_idx]
            )
        else:
            # Grammar masks follow the scheduled layout even when adaptive
            # verification compacts the actual CPU logit offsets to bonus-only.
            num_positions = int(num_draft_tokens_per_req[req_idx]) + num_bonus_tokens
        mapping.extend(
            req_idx * mask_stride + position for position in range(num_positions)
        )
    return mapping


class StructuredOutputsWorker:
    def __init__(
        self,
        max_num_logits: int,
        vocab_size: int,
        device: torch.device,
        mask_stride: int,
        num_bonus_tokens: int,
        kernels: ModelRunnerKernels,
    ):
        self.kernels = kernels
        self.logits_indices = torch.zeros(
            max_num_logits, dtype=torch.int32, device=device
        )
        self.grammar_bitmask = torch.zeros(
            (max_num_logits, cdiv(vocab_size, 32)), dtype=torch.int32, device=device
        )
        self.device = device
        self.copy_stream = torch.Stream(device=device)
        self.mask_stride = mask_stride
        self.num_bonus_tokens = num_bonus_tokens

    def apply_grammar_bitmask(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        grammar_req_ids: list[str],
        grammar_bitmask: np.ndarray,
    ) -> None:
        if not grammar_req_ids:
            return

        # Asynchronously copy the bitmask to GPU.
        current_stream = torch.accelerator.current_stream(self.device)
        with stream(self.copy_stream, current_stream):
            bitmask = async_tensor_h2d(
                grammar_bitmask, out=self.grammar_bitmask[: grammar_bitmask.shape[0]]
            )

        # Construct bitmask -> logits mapping
        # Key by (request, position) rather than absolute logit index:
        # adaptive verification finalizes per-request logit offsets on
        # device, so the kernel resolves them from the GPU cu_num_logits.
        mapping = _build_grammar_mapping(
            input_batch.req_ids,
            grammar_req_ids,
            input_batch.cu_num_logits_np,
            input_batch.num_draft_tokens_per_req,
            self.num_bonus_tokens,
            self.mask_stride,
        )

        # Asynchronously copy the mapping to GPU.
        with stream(self.copy_stream, current_stream):
            logits_indices = torch.tensor(
                mapping, dtype=torch.int32, device="cpu", pin_memory=PIN_MEMORY
            )
            logits_indices = self.logits_indices[: len(mapping)].copy_(
                logits_indices, non_blocking=True
            )

        # Ensure all async copies are complete before launching the kernel.
        current_stream.wait_stream(self.copy_stream)

        assert bitmask.shape[0] == len(mapping)
        self.kernels.apply_grammar_bitmask(
            logits, logits_indices, input_batch.cu_num_logits, bitmask, self.mask_stride
        )

        # Ensure the copy stream waits for the device tensors to finish being used
        # before it re-uses or deallocates them
        self.copy_stream.wait_stream(current_stream)

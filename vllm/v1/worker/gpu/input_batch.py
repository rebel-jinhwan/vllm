# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.utils import random_uuid
from vllm.utils.math_utils import cdiv

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.attn_utils import FastPrefillBatchMetadata
    from vllm.v1.worker.gpu.block_table import BlockTables


class InputBuffers:
    def __init__(
        self,
        max_num_reqs: int,
        max_num_tokens: int,
        device: torch.device,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_num_tokens = max_num_tokens
        self.device = device

        self.input_ids = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        self.positions = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        self.is_padding = torch.zeros(max_num_tokens, dtype=torch.bool, device=device)
        self.query_start_loc = torch.zeros(
            max_num_reqs + 1, dtype=torch.int32, device=device
        )
        self.seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        # DCP: per-request local seq_lens buffer
        self.dcp_local_seq_lens = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        )


@dataclass
class InputBatch:
    # batch_idx -> req_id
    req_ids: list[str]
    num_reqs: int
    num_reqs_after_padding: int

    # batch_idx -> req_state_idx
    idx_mapping: torch.Tensor
    idx_mapping_np: np.ndarray
    # Identical to idx_mapping except for spec decoding.
    expanded_idx_mapping: torch.Tensor
    # [total_num_logits] position within request for each logit
    expanded_local_pos: torch.Tensor

    # [num_reqs]
    # batch_idx -> num_scheduled_tokens, (upper bound when using adaptive verification)
    num_scheduled_tokens: np.ndarray
    # number of tokens in the batch,
    #  may be < sum(num_scheduled_tokens) when using adaptive verification
    num_tokens: int
    num_tokens_after_padding: int
    # Sum of draft tokens scheduled across requests.
    num_draft_tokens: int
    # [num_reqs] number of draft tokens scheduled for each request, if any.
    num_draft_tokens_per_req: np.ndarray | None

    # [num_reqs + 1]
    query_start_loc: torch.Tensor
    query_start_loc_np: np.ndarray
    # [num_reqs]
    seq_lens: torch.Tensor
    # [num_reqs] CPU upper bound on seq_lens (see CommonAttentionMetadata).
    seq_lens_cpu_upper_bound: torch.Tensor
    # [num_reqs]
    dcp_local_seq_lens: torch.Tensor | None
    # [num_reqs]
    num_computed_tokens_np: np.ndarray
    # [num_reqs]
    prefill_len_np: np.ndarray
    # [num_reqs]
    num_computed_prefill_tokens_np: np.ndarray
    # [num_reqs] CPU bool array == (num_computed_prefill_tokens_np < prefill_len_np).
    is_prefilling_np: np.ndarray
    # == np.any(is_prefilling_np)
    has_prefill: bool

    # [num_tokens_after_padding]
    input_ids: torch.Tensor
    # [num_tokens_after_padding]
    positions: torch.Tensor
    # [num_tokens_after_padding]
    is_padding: torch.Tensor

    # [total_num_logits]
    logits_indices: torch.Tensor
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor
    cu_num_logits_np: np.ndarray

    # Whether any requests in batch use structured output.
    has_structured_output_reqs: bool

    # [num_reqs] per-request prompt length, only populated for R-SWA.
    prompt_lens: torch.Tensor | None

    # Longest query the batch may contain. Set when a cudagraph descriptor promises
    # a query length this batch's own split does not reach, so attention metadata
    # stays valid for every replay the graph serves.
    max_query_len: int | None = None

    # Arms the KV-sharing fast prefill path for this step. Absent for dummy
    # (cudagraph capture) batches, which run the KV-sharing layers in full.
    fast_prefill: "FastPrefillBatchMetadata | None" = None

    # [num_reqs] set only under PCP+DCP (see CommonAttentionMetadata).
    dcp_local_seq_lens_cpu_upper_bound: torch.Tensor | None = None

    @classmethod
    def make_dummy(
        cls,
        num_reqs: int,
        num_tokens: int,
        input_buffers: InputBuffers,
        max_query_len: int | None = None,
        is_padding: bool = True,
    ) -> "InputBatch":
        assert 0 < num_reqs <= num_tokens
        device = input_buffers.device

        req_ids = [f"req_{i}_{random_uuid()}" for i in range(num_reqs)]
        idx_mapping_np = np.arange(num_reqs, dtype=np.intp)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int64, device=device)
        expanded_idx_mapping = idx_mapping
        expanded_local_pos = torch.zeros(num_reqs, dtype=torch.int32, device=device)

        # Distribute the remainder evenly so that no dummy request exceeds
        # ceil(num_tokens / num_reqs) <= max_model_len tokens. Varlen graphs
        # accept any split with non-empty slots, so this shape works for them
        # too; attention metadata is built from the promised max_query_len.
        base_tokens = num_tokens // num_reqs
        num_extra = num_tokens % num_reqs
        assert max_query_len is None or base_tokens + (num_extra > 0) <= max_query_len
        num_scheduled_tokens = np.full(num_reqs, base_tokens, dtype=np.int32)
        if num_extra > 0:
            num_scheduled_tokens[-num_extra:] += 1
        assert int(num_scheduled_tokens.sum()) == num_tokens

        # seq_len equals to query_len
        input_buffers.seq_lens[: num_reqs - num_extra] = base_tokens
        input_buffers.seq_lens[num_reqs - num_extra : num_reqs] = base_tokens + 1
        # Pad for full CUDA graph mode.
        input_buffers.seq_lens[num_reqs:] = 0
        seq_lens = input_buffers.seq_lens[:num_reqs]

        query_start_loc_np = np.empty(num_reqs + 1, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1:])
        input_buffers.query_start_loc[:1] = 0
        torch.cumsum(
            seq_lens, dim=0, out=input_buffers.query_start_loc[1 : num_reqs + 1]
        )
        # Pad for full CUDA graph mode.
        input_buffers.query_start_loc[num_reqs + 1 :] = num_tokens
        query_start_loc = input_buffers.query_start_loc[: num_reqs + 1]

        input_ids = input_buffers.input_ids[:num_tokens].zero_()
        positions = input_buffers.positions[:num_tokens].zero_()

        input_buffers.is_padding[:num_tokens].fill_(is_padding)
        is_padding = input_buffers.is_padding[:num_tokens]

        logits_indices = query_start_loc[1:] - 1
        cu_num_logits = torch.arange(num_reqs + 1, device=device, dtype=torch.int32)
        cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
        # Copy so set_dummy_context can add context in place without touching
        # num_scheduled_tokens.
        seq_lens_cpu_upper_bound = torch.from_numpy(num_scheduled_tokens.copy())
        return cls(
            req_ids=req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens,
            num_draft_tokens=0,
            num_draft_tokens_per_req=None,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=None,
            num_computed_tokens_np=np.zeros(num_reqs, dtype=np.int32),
            prefill_len_np=np.zeros(num_reqs, dtype=np.int32),
            num_computed_prefill_tokens_np=np.zeros(num_reqs, dtype=np.int32),
            is_prefilling_np=np.zeros(num_reqs, dtype=np.bool_),
            has_prefill=False,
            input_ids=input_ids,
            positions=positions,
            is_padding=is_padding,
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=False,
            prompt_lens=None,
            max_query_len=max_query_len,
        )


def set_dummy_context(
    input_batch: InputBatch,
    block_tables: "BlockTables",
    context_len: int,
    num_kv_blocks: int,
    max_model_len: int,
    input_block_tables: Sequence[torch.Tensor] | None = None,
) -> None:
    """Give each dummy request context_len of context, used when profiling step cost."""
    if input_block_tables is None:
        input_block_tables = block_tables.input_block_tables
    if not input_block_tables:
        # Attention-free models have no KV context to fabricate.
        return
    num_reqs = input_batch.num_reqs
    query_len = input_batch.max_query_len or int(input_batch.num_scheduled_tokens.max())
    context_len = max(min(context_len, max_model_len - query_len), 0)
    if not context_len:
        return

    # Decode-like shape: each request continues after context_len
    # already-computed tokens.
    input_batch.seq_lens += context_len
    input_batch.seq_lens_cpu_upper_bound += context_len
    input_batch.num_computed_tokens_np.fill(context_len)
    input_batch.num_computed_prefill_tokens_np.fill(context_len)
    local_pos = np.arange(input_batch.num_tokens, dtype=np.int64) - np.repeat(
        input_batch.query_start_loc_np[:-1], input_batch.num_scheduled_tokens
    )
    input_batch.positions.copy_(torch.from_numpy(local_pos + context_len))

    seq_len = context_len + query_len
    for block_table, block_size, bpk in zip(
        input_block_tables,
        block_tables.kernel_block_sizes,
        block_tables.blocks_per_kv_block,
    ):
        num_blocks = min(cdiv(seq_len, block_size), block_table.shape[1])
        # Spans are disjoint until the pool runs out, then they wrap and share
        # blocks: profiling only needs the reads to be realistic, not distinct.
        block_ids = torch.arange(
            num_reqs * num_blocks, dtype=block_table.dtype, device=block_table.device
        ) % (num_kv_blocks * bpk)
        block_table[:num_reqs, :num_blocks] = block_ids.view(num_reqs, num_blocks)

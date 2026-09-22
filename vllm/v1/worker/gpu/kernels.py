# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`ModelRunnerKernels` on Triton. The kernels the V2 runner, its sampler stack
and the draft-model speculators launch live here; `rejection_sample` keeps its
own module."""

from typing import TYPE_CHECKING

import torch
from torch._inductor.runtime.triton_helpers import libdevice

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_block_argmax
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample
from vllm.v1.worker.kernels import ModelRunnerKernels

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers


@triton.jit
def _prepare_prefill_inputs_kernel(
    input_ids_ptr,
    next_prefill_tokens_ptr,
    next_prefill_tokens_stride,
    num_lookahead,
    idx_mapping_ptr,
    query_start_loc_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    prefill_lens_ptr,
    num_computed_tokens_ptr,
    BLOCK_SIZE: tl.constexpr,
    LOOKAHEAD_BLOCK: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    prefill_len = tl.load(prefill_lens_ptr + req_state_idx)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    if num_computed >= prefill_len:
        # Not prefill.
        return

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    request_ptr = all_token_ids_ptr + req_state_idx * all_token_ids_stride
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        tokens = tl.load(request_ptr + num_computed + block, mask=mask)
        tl.store(input_ids_ptr + query_start + block, tokens, mask=mask)

    # Store the next num_lookahead prefill tokens.
    lookahead = tl.arange(0, LOOKAHEAD_BLOCK)
    pos = num_computed + query_len + lookahead
    in_lookahead = lookahead < num_lookahead
    tokens = tl.load(
        request_ptr + pos, mask=in_lookahead & (pos < prefill_len), other=0
    )
    tl.store(
        next_prefill_tokens_ptr
        + lookahead * next_prefill_tokens_stride
        + req_state_idx,
        tokens,
        mask=in_lookahead,
    )


@triton.jit
def _prepare_pos_seq_lens_kernel(
    pos_ptr,
    seq_lens_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    num_computed_tokens_ptr,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    req_id = tl.program_id(0)
    num_reqs = tl.num_programs(0) - 1
    if req_id == num_reqs:
        # Pad unused seq_lens as 0 for full CUDA graphs.
        for i in tl.range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(seq_lens_ptr + block, 0, mask=mask)
        return

    req_state_idx = tl.load(idx_mapping_ptr + req_id)
    num_computed_tokens = tl.load(num_computed_tokens_ptr + req_state_idx)

    start = tl.load(query_start_loc_ptr + req_id)
    end = tl.load(query_start_loc_ptr + req_id + 1)
    query_len = end - start

    seq_len = num_computed_tokens + query_len
    tl.store(seq_lens_ptr + req_id, seq_len)

    for i in tl.range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        pos = num_computed_tokens + block
        tl.store(pos_ptr + start + block, pos, mask=mask)


@triton.jit
def _combine_sampled_and_draft_tokens_kernel(
    input_ids_ptr,
    idx_mapping_ptr,
    last_sampled_tokens_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    prefill_len_ptr,
    draft_tokens_ptr,
    draft_tokens_stride,
    cu_num_logits_ptr,
    logits_indices_ptr,
    BLOCK_SIZE: tl.constexpr,
    NUM_NEW_SAMPLED_TOKENS: tl.constexpr = 1,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    # Get the number of logits and draft tokens.
    cu_num_logits_start = tl.load(cu_num_logits_ptr + batch_idx)
    cu_num_logits_end = tl.load(cu_num_logits_ptr + batch_idx + 1)
    num_logits = cu_num_logits_end - cu_num_logits_start
    num_draft_tokens = num_logits - NUM_NEW_SAMPLED_TOKENS

    # Compute the logits indices.
    block = tl.arange(0, BLOCK_SIZE)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    logits_start = query_end - num_logits
    tl.store(
        logits_indices_ptr + cu_num_logits_start + block,
        logits_start + block,
        mask=block < num_logits,
    )

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    if seq_len <= prefill_len:
        # Handling prefill tokens. No sampled or draft tokens.
        return

    # Keep prompt-tail slots intact; only rewrite generated-token slots.
    first_logit_seq_pos = seq_len - num_logits
    if NUM_NEW_SAMPLED_TOKENS > 0 and first_logit_seq_pos >= prefill_len:
        # Write the last sampled token ID to input_ids.
        last_token_id = tl.load(last_sampled_tokens_ptr + req_state_idx)
        tl.store(input_ids_ptr + logits_start, last_token_id)

    # Write the draft tokens (if any) to input_ids.
    if num_draft_tokens > 0:
        mask = block < num_draft_tokens
        draft_tokens = tl.load(
            draft_tokens_ptr + req_state_idx * draft_tokens_stride + block,
            mask=mask,
        )
        tl.store(
            input_ids_ptr + query_end - num_draft_tokens + block,
            draft_tokens,
            mask=mask,
        )


@triton.jit
def _get_num_sampled_and_rejected_kernel(
    num_sampled_ptr,
    num_rejected_ptr,
    seq_lens_ptr,
    cu_num_logits_ptr,
    idx_mapping_ptr,
    prefill_len_ptr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    is_chunked_prefilling = seq_len < prefill_len

    num_sampled = tl.load(num_sampled_ptr + batch_idx)
    num_sampled = tl.where(is_chunked_prefilling, 0, num_sampled)
    tl.store(num_sampled_ptr + batch_idx, num_sampled)

    logits_start = tl.load(cu_num_logits_ptr + batch_idx)
    logits_end = tl.load(cu_num_logits_ptr + batch_idx + 1)
    num_logits = logits_end - logits_start

    num_rejected = num_logits - num_sampled
    num_rejected = tl.where(is_chunked_prefilling, 0, num_rejected)
    tl.store(num_rejected_ptr + batch_idx, num_rejected)


@triton.jit
def _post_update_kernel(
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    last_sampled_tokens_ptr,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    sampled_tokens_ptr,
    sampled_tokens_stride,
    num_sampled_ptr,
    num_rejected_ptr,
    query_start_loc_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
):
    req_id = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_id)
    if req_state_idx < 0:
        # Filter rows with negative index entries.
        return

    total_len = tl.load(total_len_ptr + req_state_idx)
    num_sampled = tl.load(num_sampled_ptr + req_id)
    if num_sampled > 0:
        token_id = tl.load(
            sampled_tokens_ptr + req_id * sampled_tokens_stride + num_sampled - 1
        )
        tl.store(last_sampled_tokens_ptr + req_state_idx, token_id)
        tl.store(total_len_ptr + req_state_idx, total_len + num_sampled)

    for i in range(num_sampled):
        token_id = tl.load(sampled_tokens_ptr + req_id * sampled_tokens_stride + i)
        tl.store(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + total_len + i,
            token_id,
        )

        if output_bin_counts_ptr is not None:
            token_ptr = (
                output_bin_counts_ptr
                + req_state_idx * output_bin_counts_stride
                + token_id
            )
            count = tl.load(token_ptr)
            tl.store(token_ptr, count + 1)

    if query_start_loc_ptr is None:
        query_len = 0
    else:
        query_start = tl.load(query_start_loc_ptr + req_id)
        query_end = tl.load(query_start_loc_ptr + req_id + 1)
        query_len = query_end - query_start
    num_rejected = tl.load(num_rejected_ptr + req_id)

    computed_delta = query_len - num_rejected
    if computed_delta != 0:
        num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
        tl.store(num_computed_tokens_ptr + req_state_idx, num_computed + computed_delta)


@triton.jit
def _post_update_num_computed_tokens_kernel(
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    query_start_loc_ptr,
):
    batch_id = tl.program_id(0)
    query_start = tl.load(query_start_loc_ptr + batch_id)
    query_end = tl.load(query_start_loc_ptr + batch_id + 1)
    query_len = query_end - query_start

    req_state_idx = tl.load(idx_mapping_ptr + batch_id)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    tl.store(num_computed_tokens_ptr + req_state_idx, num_computed + query_len)


@triton.jit
def _expand_idx_mapping_kernel(
    idx_mapping_ptr,
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    cu_num_logits_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_tokens = end_idx - start_idx

    block = tl.arange(0, BLOCK_SIZE)
    mask = block < num_tokens
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    tl.store(expanded_idx_mapping_ptr + start_idx + block, req_state_idx, mask=mask)
    tl.store(expanded_local_pos_ptr + start_idx + block, block, mask=mask)


@triton.jit
def _num_nans_kernel(
    logits_ptr,
    logits_stride,
    num_nans_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_nans = 0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(
            logits_ptr + req_idx * logits_stride + block, mask=mask, other=0
        )
        logits = logits.to(tl.float32)
        is_nan = libdevice.isnan(logits).to(tl.int1)
        num_nans += tl.sum(is_nan).to(tl.int32)
    tl.store(num_nans_ptr + req_idx, num_nans)


@triton.jit
def _temperature_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    temperature_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    if temperature == 0.0 or temperature == 1.0:
        # Early return to avoid loading logits.
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    logits = tl.load(logits_ptr + token_idx * logits_stride + block, mask=mask)
    logits = logits.to(tl.float32)
    logits = logits / temperature
    tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


@triton.jit
def _gumbel_sample_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    # [max_num_reqs, num_cols, vocab_size]
    logits_cache_ptr,
    logits_cache_stride_0,
    logits_cache_stride_1,
    logits_cache_col_ptr,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    IS_DRAFTING: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    USE_FP64: tl.constexpr,
    PER_TOKEN_COL: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        logits_ptr + token_idx * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    logits = logits.to(tl.float32)

    value, idx = gumbel_block_argmax(
        logits,
        block,
        mask,
        token_idx,
        expanded_idx_mapping_ptr,
        temp_ptr,
        seeds_ptr,
        pos_ptr,
        logits_cache_ptr,
        logits_cache_stride_0,
        logits_cache_stride_1,
        logits_cache_col_ptr,
        vocab_size,
        IS_DRAFTING=IS_DRAFTING,
        APPLY_TEMPERATURE=APPLY_TEMPERATURE,
        USE_FP64=USE_FP64,
        PER_TOKEN_COL=PER_TOKEN_COL,
    )
    token_id = block_idx * BLOCK_SIZE + idx
    tl.store(local_argmax_ptr + token_idx * local_argmax_stride + block_idx, token_id)
    tl.store(local_max_ptr + token_idx * local_max_stride + block_idx, value)


def gumbel_sample(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    expanded_idx_mapping: torch.Tensor,  # [num_tokens]
    temperature: torch.Tensor,  # [max_num_reqs]
    seed: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [num_tokens]
    apply_temperature: bool,
    is_drafting: bool,
    logits_cache: torch.Tensor | None = None,  # [max_num_reqs, num_cols, vocab_size]
    logits_cache_col: torch.Tensor | None = None,  # scalar or [num_tokens]
    use_fp64: bool = False,
) -> torch.Tensor:
    # Enforce contiguity on non-strided input tensors
    expanded_idx_mapping = expanded_idx_mapping.contiguous()
    pos = pos.contiguous()
    if logits_cache_col is not None:
        logits_cache_col = logits_cache_col.contiguous()
    num_tokens, vocab_size = logits.shape
    if logits_cache is not None:
        assert logits_cache.size(-1) >= vocab_size, (
            f"draft logits cache vocab dim ({logits_cache.size(-1)}) is narrower "
            f"than the sampled logits ({vocab_size}). Cached logits would be "
            "truncated."
        )
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
    local_max_dtype = torch.float64 if use_fp64 else torch.float32
    local_max = logits.new_empty(num_tokens, num_blocks, dtype=local_max_dtype)
    per_token_col = logits_cache_col is not None and logits_cache_col.dim() > 0
    _gumbel_sample_kernel[(num_tokens, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        logits_cache,
        logits_cache.stride(0) if logits_cache is not None else 0,
        logits_cache.stride(1) if logits_cache is not None else 0,
        logits_cache_col,
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        seed,
        pos,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        IS_DRAFTING=is_drafting,
        APPLY_TEMPERATURE=apply_temperature,
        USE_FP64=use_fp64,
        PER_TOKEN_COL=per_token_col,
    )
    # NOTE(woosuk): Use int64 for later indexing.
    max_block_idx = local_max.argmax(dim=-1, keepdim=True)
    sampled = local_argmax.gather(dim=-1, index=max_block_idx).view(-1)
    return sampled


@triton.jit
def _min_p_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    min_p_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    min_p = tl.load(min_p_ptr + req_state_idx).to(tl.float32)
    if min_p == 0.0:
        return

    max_val = float("-inf")
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(
            logits_ptr + token_idx * logits_stride + block,
            mask=mask,
            other=float("-inf"),
        )
        max_val = tl.max(tl.maximum(logits, max_val))
    max_val = max_val.to(tl.float32)  # type: ignore

    threshold = max_val + tl.log(min_p)
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(
            logits_ptr + token_idx * logits_stride + block,
            mask=mask,
            other=float("-inf"),
        )
        logits = tl.where(logits < threshold, float("-inf"), logits)
        tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


@triton.jit
def _penalties_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    token_ids_ptr,
    expanded_local_pos_ptr,
    repetition_penalty_ptr,
    frequency_penalty_ptr,
    presence_penalty_ptr,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    rep_penalty = tl.load(repetition_penalty_ptr + req_state_idx)
    freq_penalty = tl.load(frequency_penalty_ptr + req_state_idx)
    pres_penalty = tl.load(presence_penalty_ptr + req_state_idx)

    use_rep_penalty = rep_penalty != 1.0
    use_freq_penalty = freq_penalty != 0.0
    use_pres_penalty = pres_penalty != 0.0
    use_penalty = use_rep_penalty or use_freq_penalty or use_pres_penalty
    if not use_penalty:
        # Early return to avoid loading logits.
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(logits_ptr + token_idx * logits_stride + block, mask=mask)
    logits = logits.to(tl.float32)

    base_output_counts = tl.load(
        output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + block,
        mask=mask,
        other=0,
    )

    # Accumulate draft token counts from previous positions directly into
    # output_bin_counts (preserves its native tensor layout, avoiding an
    # expensive shared-memory layout conversion after the loop).
    pos = tl.load(expanded_local_pos_ptr + token_idx)
    start_idx = token_idx - pos
    output_bin_counts = base_output_counts
    for prev_pos in tl.range(pos):
        prev_token = tl.load(token_ids_ptr + start_idx + prev_pos + 1)
        token_match = block == prev_token
        output_bin_counts = output_bin_counts + token_match.to(tl.int32)
    output_bin_mask = output_bin_counts > 0

    # Apply repetition penalties.
    if use_rep_penalty:
        packed_block = block_idx * BLOCK_SIZE // 32 + tl.arange(0, BLOCK_SIZE // 32)
        packed_mask = tl.load(
            prompt_bin_mask_ptr + req_state_idx * prompt_bin_mask_stride + packed_block,
            mask=packed_block < tl.cdiv(vocab_size, 32),
            other=0,
        )
        prompt_bin_mask = (packed_mask[:, None] >> (tl.arange(0, 32)[None, :])) & 1
        prompt_bin_mask = prompt_bin_mask.to(tl.int1)
        prompt_bin_mask = prompt_bin_mask.reshape(BLOCK_SIZE)

        # If token appears in prompt or output, apply, otherwise use 1.0 for no-op.
        scale = tl.where(prompt_bin_mask | output_bin_mask, rep_penalty, 1.0)
        # If logits are positive, divide by penalty, otherwise multiply by penalty.
        logits *= tl.where(logits > 0, 1.0 / scale, scale)

    # Apply frequency penalties.
    logits -= freq_penalty * output_bin_counts
    # Apply presence penalties.
    logits -= pres_penalty * output_bin_mask
    # Store back to logits.
    tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


@triton.jit
def _bincount_kernel(
    expanded_idx_mapping_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    prompt_len_ptr,
    prefill_len_ptr,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)

    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    if block_idx * BLOCK_SIZE >= prefill_len:
        return

    prompt_len = tl.load(prompt_len_ptr + req_state_idx)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if block_idx * BLOCK_SIZE < prompt_len:
        mask = block < prompt_len
        prompt_tokens = tl.load(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + block, mask=mask
        )
        idx = prompt_tokens // 32
        bit_idx = prompt_tokens % 32
        bit = tl.full((BLOCK_SIZE,), 1, tl.int32) << bit_idx
        tl.atomic_or(
            prompt_bin_mask_ptr + req_state_idx * prompt_bin_mask_stride + idx,
            bit,
            mask=mask,
        )

    if (block_idx + 1) * BLOCK_SIZE >= prompt_len:
        mask = block < prefill_len
        mask &= block >= prompt_len
        output_tokens = tl.load(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + block, mask=mask
        )
        tl.atomic_add(
            output_bin_counts_ptr
            + req_state_idx * output_bin_counts_stride
            + output_tokens,
            1,
            mask=mask,
        )


@triton.jit
def _bias_kernel(
    logits_ptr,
    logits_stride,
    vocab_size,
    expanded_idx_mapping_ptr,
    # Allowed token IDs.
    num_allowed_token_ids_ptr,
    allowed_token_ids_ptr,
    allowed_token_ids_stride,
    # Logit bias.
    num_logit_bias_ptr,
    bias_token_ids_ptr,
    bias_token_ids_stride,
    bias_ptr,
    bias_stride,
    # Min tokens.
    pos_ptr,
    min_lens_ptr,
    num_stop_token_ids_ptr,
    restore_when_all_masked_ptr,
    stop_token_ids_ptr,
    stop_token_ids_stride,
    BLOCK_SIZE: tl.constexpr,
    LOGITS_BLOCK_SIZE: tl.constexpr,
    CHECK_ALL_MASKED_ROWS: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)

    block = tl.arange(0, BLOCK_SIZE)

    # Allowed token IDs.
    num_allowed_token_ids = tl.load(num_allowed_token_ids_ptr + req_state_idx)
    if num_allowed_token_ids > 0:
        block = tl.arange(0, BLOCK_SIZE)
        mask = block < num_allowed_token_ids

        # Save logits for allowed token IDs.
        allowed_token_ids = tl.load(
            allowed_token_ids_ptr + req_state_idx * allowed_token_ids_stride + block,
            mask=mask,
        )
        logits = tl.load(
            logits_ptr + token_idx * logits_stride + allowed_token_ids, mask=mask
        )

        tl.debug_barrier()  # save must read original logits before the -inf overwrite

        # Set logits to -inf for all tokens.
        for i in range(0, vocab_size, LOGITS_BLOCK_SIZE):
            offset = i + tl.arange(0, LOGITS_BLOCK_SIZE)
            tl.store(
                logits_ptr + token_idx * logits_stride + offset,
                -float("inf"),
                mask=offset < vocab_size,
            )

        tl.debug_barrier()  # -inf overwrite must finish before restoring saved logits

        # Restore logits for allowed token IDs.
        tl.store(
            logits_ptr + token_idx * logits_stride + allowed_token_ids,
            logits,
            mask=mask,
        )

    # Logit bias.
    num_logit_bias = tl.load(num_logit_bias_ptr + req_state_idx)
    if num_logit_bias > 0:
        mask = block < num_logit_bias
        token_ids = tl.load(
            bias_token_ids_ptr + req_state_idx * bias_token_ids_stride + block,
            mask=mask,
        )
        bias = tl.load(bias_ptr + req_state_idx * bias_stride + block, mask=mask)
        logits = tl.load(logits_ptr + token_idx * logits_stride + token_ids, mask=mask)
        logits += bias
        tl.store(logits_ptr + token_idx * logits_stride + token_ids, logits, mask=mask)

    # Apply min tokens.
    num_stop_token_ids = tl.load(num_stop_token_ids_ptr + req_state_idx)
    pos = tl.load(pos_ptr + token_idx)
    min_len = tl.load(min_lens_ptr + req_state_idx)
    if num_stop_token_ids > 0 and pos + 1 < min_len:
        mask = block < num_stop_token_ids
        stop_token_ids = tl.load(
            stop_token_ids_ptr + req_state_idx * stop_token_ids_stride + block,
            mask=mask,
        )
        if CHECK_ALL_MASKED_ROWS:
            should_restore_stop_logits = tl.load(
                restore_when_all_masked_ptr + req_state_idx
            )
            if should_restore_stop_logits:
                stop_logits = tl.load(
                    logits_ptr + token_idx * logits_stride + stop_token_ids,
                    mask=mask,
                    other=-float("inf"),
                )

                # Save must read original logits before the -inf overwrite.
                tl.debug_barrier()

                tl.store(
                    logits_ptr + token_idx * logits_stride + stop_token_ids,
                    -float("inf"),
                    mask=mask,
                )

                # The -inf overwrite must be visible to the row scan below.
                tl.debug_barrier()

                row_max = tl.full((), -float("inf"), tl.float32)
                for i in range(0, vocab_size, LOGITS_BLOCK_SIZE):
                    offset = i + tl.arange(0, LOGITS_BLOCK_SIZE)
                    logits = tl.load(
                        logits_ptr + token_idx * logits_stride + offset,
                        mask=offset < vocab_size,
                        other=-float("inf"),
                    )
                    row_max = tl.maximum(row_max, tl.max(logits, axis=0))

                if row_max == -float("inf"):
                    tl.store(
                        logits_ptr + token_idx * logits_stride + stop_token_ids,
                        stop_logits,
                        mask=mask
                        & (stop_logits > -float("inf"))
                        & (stop_logits < float("inf")),
                    )
            else:
                tl.store(
                    logits_ptr + token_idx * logits_stride + stop_token_ids,
                    -float("inf"),
                    mask=mask,
                )
        else:
            tl.store(
                logits_ptr + token_idx * logits_stride + stop_token_ids,
                -float("inf"),
                mask=mask,
            )


@triton.jit
def _bad_words_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    bad_word_token_ids_ptr,
    bad_word_token_ids_stride,
    bad_word_offsets_ptr,
    bad_word_offsets_stride,
    num_bad_words_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    prompt_len_ptr,
    total_len_ptr,
    input_ids_ptr,
    expanded_local_pos_ptr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    bw_idx = tl.program_id(1)

    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    num_bad_words = tl.load(num_bad_words_ptr + req_state_idx)

    if bw_idx >= num_bad_words:
        return

    pos = tl.load(expanded_local_pos_ptr + token_idx)
    cur_req_first_pos = token_idx - pos

    prompt_len = tl.load(prompt_len_ptr + req_state_idx)
    total_len = tl.load(total_len_ptr + req_state_idx)
    output_len = total_len - prompt_len
    effective_len = output_len + pos

    bd_offsets_base = bad_word_offsets_ptr + req_state_idx * bad_word_offsets_stride
    bd_tokens_base = bad_word_token_ids_ptr + req_state_idx * bad_word_token_ids_stride
    output_base = all_token_ids_ptr + req_state_idx * all_token_ids_stride + prompt_len

    start = tl.load(bd_offsets_base + bw_idx)
    end = tl.load(bd_offsets_base + bw_idx + 1)
    bad_word_len = end - start
    prefix_len = bad_word_len - 1

    if prefix_len > effective_len:
        return

    last_token = tl.load(bd_tokens_base + end - 1)
    match = 1
    for i in range(prefix_len):
        expected = tl.load(bd_tokens_base + start + i)
        actual_pos = effective_len - prefix_len + i

        from_spec_input = actual_pos >= output_len
        if from_spec_input:
            # input_ids at local position 0 is the last committed token;
            # draft tokens start at local position 1.
            spec_offset = actual_pos - output_len
            actual = tl.load(input_ids_ptr + cur_req_first_pos + spec_offset + 1)
        else:
            actual = tl.load(output_base + actual_pos)

        match = match & (expected == actual)

    if match:
        tl.store(logits_ptr + token_idx * logits_stride + last_token, -float("inf"))


# Upper bound on the topk kernel's per-iteration gather width.
_MAX_TOPK_BLOCK = 1024


@triton.jit
def _topk_log_softmax_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    topk_ids_ptr,
    topk,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    TOPK_BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + req_idx * logits_stride

    max_val = float("-inf")
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
        max_val = tl.max(tl.maximum(logits, max_val))
    max_val = max_val.to(tl.float32)  # type: ignore

    se = 0.0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=0.0)
        # NOTE(woosuk): Make sure that logits and all following operations use FP32.
        logits = logits.to(tl.float32)
        e = tl.exp(logits - max_val)
        e = tl.where(block < vocab_size, e, 0.0)
        se += tl.sum(e)
    lse = tl.log(se)

    for j in range(0, topk, TOPK_BLOCK_SIZE):
        k_offset = j + tl.arange(0, TOPK_BLOCK_SIZE)
        k_mask = k_offset < topk
        topk_ids = tl.load(
            topk_ids_ptr + req_idx * topk + k_offset, mask=k_mask, other=0
        )
        logits = tl.load(row_ptr + topk_ids, mask=k_mask)
        logits = logits.to(tl.float32)
        o = logits - max_val - lse
        tl.store(output_ptr + req_idx * topk + k_offset, o, mask=k_mask)


@triton.jit
def _ranks_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    token_ids_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + req_idx * logits_stride

    token_id = tl.load(token_ids_ptr + req_idx)
    x = tl.load(row_ptr + token_id)

    n = 0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
        n += tl.sum((logits >= x).to(tl.int32))
    tl.store(output_ptr + req_idx, n)


def compute_token_logprobs(
    logits: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    # NOTE(woosuk): To save GPU memory, we do not materialize the full
    # [batch_size, vocab_size] logprobs tensor. The kernel computes
    # max + logsumexp per row and only emits logprobs at `token_ids`.
    batch_size, vocab_size = logits.shape
    token_ids = token_ids.to(torch.int64)
    num_logprobs = token_ids.shape[1]
    logprobs = logits.new_empty((batch_size, num_logprobs), dtype=torch.float32)
    # Cap the kernel's per-iteration width so very large num_logprobs requests
    # stream the gather in bounded-size chunks, avoiding excessive mem use.
    topk_block_size = min(triton.next_power_of_2(num_logprobs), _MAX_TOPK_BLOCK)
    _topk_log_softmax_kernel[(batch_size,)](
        logprobs,
        logits,
        logits.stride(0),
        token_ids,
        num_logprobs,
        vocab_size,
        BLOCK_SIZE=1024,  # type: ignore
        TOPK_BLOCK_SIZE=topk_block_size,
    )
    return logprobs


@triton.jit
def _fill_logprob_token_ids_kernel(
    # [batch_size, 1 + num_cols]
    out_token_ids_ptr,
    out_token_ids_stride,
    # [batch_size, 1 + num_cols]
    out_valid_mask_ptr,
    out_valid_mask_stride,
    sampled_token_ids_ptr,  # [batch_size]
    topk_indices_ptr,  # [batch_size, NUM_TOPK] (unused when NUM_TOPK == 0)
    topk_indices_stride,
    expanded_idx_mapping_ptr,  # [batch_size] -> req_state_idx
    num_per_req_token_ids_ptr,  # [max_num_reqs]
    per_req_token_ids_ptr,  # [max_num_reqs, MAX_LOGPROB_TOKEN_IDS]
    per_req_token_ids_stride,
    NUM_TOPK: tl.constexpr,
    PADDED_COLS: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    # Column 0: always the sampled token, always valid.
    sampled = tl.load(sampled_token_ids_ptr + batch_idx)
    tl.store(out_token_ids_ptr + batch_idx * out_token_ids_stride, sampled)
    tl.store(out_valid_mask_ptr + batch_idx * out_valid_mask_stride, 1)

    req_state_idx = tl.load(expanded_idx_mapping_ptr + batch_idx)
    num_custom = tl.load(num_per_req_token_ids_ptr + req_state_idx)

    col = tl.arange(0, PADDED_COLS)
    tid_base = out_token_ids_ptr + batch_idx * out_token_ids_stride + 1
    mask_base = out_valid_mask_ptr + batch_idx * out_valid_mask_stride + 1

    if num_custom > 0:
        # Override topk with per-request custom tokens.
        src = per_req_token_ids_ptr + req_state_idx * per_req_token_ids_stride
        valid = col < num_custom
    else:
        # Fill with topk indices (no-op when NUM_TOPK == 0).
        src = topk_indices_ptr + batch_idx * topk_indices_stride
        valid = col < NUM_TOPK

    tokens = tl.load(src + col, mask=valid, other=0).to(tl.int64)
    tl.store(tid_base + col, tokens, mask=valid)
    tl.store(mask_base + col, tl.full([PADDED_COLS], 1, tl.int1), mask=valid)


@triton.jit
def _prompt_logprobs_token_ids_kernel(
    prompt_logprobs_token_ids_ptr,
    query_start_loc_ptr,
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    num_computed_tokens = tl.load(num_computed_tokens_ptr + req_state_idx)
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        # NOTE(woosuk): We should shift the pos by one
        # because the logprob is computed for the next token.
        target_pos = num_computed_tokens + 1 + block
        token_ids = tl.load(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + target_pos,
            mask=mask,
        )
        tl.store(
            prompt_logprobs_token_ids_ptr + query_start + block, token_ids, mask=mask
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


# Adapted from
# https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/kernels/apply_token_bitmask_inplace_triton.py
@triton.jit
def _apply_grammar_bitmask_kernel(
    logits_ptr,
    logits_stride,
    logits_indices_ptr,
    cu_num_logits_ptr,
    bitmask_ptr,
    bitmask_stride,
    vocab_size,
    MASK_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    bitmask_idx = tl.program_id(0)
    mapping_idx = tl.load(logits_indices_ptr + bitmask_idx)
    req_idx = mapping_idx // MASK_STRIDE
    position_idx = mapping_idx % MASK_STRIDE
    logits_idx = tl.load(cu_num_logits_ptr + req_idx)
    num_req_logits = tl.load(cu_num_logits_ptr + req_idx + 1) - logits_idx
    logits_idx += position_idx
    position_is_active = position_idx < num_req_logits

    # Load the bitmask.
    block_id = tl.program_id(1)
    bitmask_offset = (block_id * BLOCK_SIZE) // 32 + tl.arange(0, BLOCK_SIZE // 32)
    packed_bitmask = tl.load(
        bitmask_ptr + bitmask_idx * bitmask_stride + bitmask_offset,
        mask=bitmask_offset < bitmask_stride,
    )
    # Unpack the bitmask.
    bitmask = ((packed_bitmask[:, None] >> (tl.arange(0, 32)[None, :])) & 1) == 0
    bitmask = bitmask.reshape(BLOCK_SIZE)

    # Apply the bitmask to the logits.
    block_offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(
        logits_ptr + logits_idx * logits_stride + block_offset,
        -float("inf"),
        mask=position_is_active & bitmask & (block_offset < vocab_size),
    )


@triton.jit
def _prepare_draft_prefill_inputs_kernel(
    last_token_indices_ptr,
    draft_current_step_ptr,
    draft_input_ids_ptr,
    draft_positions_ptr,
    draft_query_start_loc_ptr,
    draft_seq_lens_ptr,
    target_input_ids_ptr,
    target_positions_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + req_idx)

    # Get the true query length and next token after accounting for rejected tokens.
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    query_len -= num_rejected

    num_sampled = tl.load(num_sampled_ptr + req_idx)
    if num_sampled > 0:
        next_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        # Chunked prefilling.
        # Get the next prefill token.
        next_token = tl.load(next_prefill_tokens_ptr + req_state_idx)

    # Shift target_input_ids by one.
    for i in range(1, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        input_ids = tl.load(target_input_ids_ptr + query_start + block, mask=mask)
        tl.store(draft_input_ids_ptr + query_start + block - 1, input_ids, mask=mask)

    last_token_index = query_start + query_len - 1
    tl.store(last_token_indices_ptr + req_idx, last_token_index)
    tl.store(draft_input_ids_ptr + last_token_index, next_token)

    # Copy positions.
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        target_pos = tl.load(target_positions_ptr + query_start + block, mask=mask)
        tl.store(draft_positions_ptr + query_start + block, target_pos, mask=mask)

    # Copy query start locations.
    tl.store(draft_query_start_loc_ptr + req_idx, query_start)
    # Copy sequence lengths.
    tl.store(draft_seq_lens_ptr + req_idx, seq_len)
    if req_idx == (num_reqs - 1):
        # Reset the current draft step to 0.
        tl.store(draft_current_step_ptr, 0)
        # Pad query_start_loc for CUDA graphs.
        for i in range(num_reqs, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs + 1
            tl.store(draft_query_start_loc_ptr + block, query_end, mask=mask)
        # Pad seq_lens for CUDA graphs.
        for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(draft_seq_lens_ptr + block, 0, mask=mask)
        # Pad last_token_indices for CUDA graphs.
        for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(last_token_indices_ptr + block, 0, mask=mask)


@triton.jit
def _prepare_draft_decode_inputs_kernel(
    draft_tokens_ptr,
    draft_tokens_stride,
    target_seq_lens_ptr,
    num_rejected_ptr,
    input_ids_ptr,
    positions_ptr,
    sample_src_positions_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    max_model_len,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
    ADVANCE_DRAFT_POSITIONS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0) - 1
    if req_idx == num_reqs:
        # Compute query_start_loc. Pad it with the last query_start_loc
        # for CUDA graphs.
        for i in range(0, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            q = tl.where(block < num_reqs, block, num_reqs)
            mask = block < max_num_reqs + 1
            tl.store(query_start_loc_ptr + block, q, mask=mask)
        # Pad seq_lens for CUDA graphs.
        for i in range(req_idx, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(seq_lens_ptr + block, 0, mask=mask)
        return

    # draft token -> input id.
    draft_token = tl.load(draft_tokens_ptr + req_idx * draft_tokens_stride)
    tl.store(input_ids_ptr + req_idx, draft_token)

    # Advance the draft sampling key.
    sample_position = tl.load(sample_src_positions_ptr + req_idx)
    tl.store(sample_src_positions_ptr + req_idx, sample_position + 1)

    target_seq_len = tl.load(target_seq_lens_ptr + req_idx)
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    seq_len = target_seq_len - num_rejected
    if ADVANCE_DRAFT_POSITIONS:
        # Compute position and seq_lens.
        # NOTE(woosuk): To prevent out-of-range access, we clamp these values
        # if they reach the max model length.
        position = tl.load(positions_ptr + req_idx)
        position = tl.minimum(position + 1, max_model_len - 1)
        tl.store(positions_ptr + req_idx, position)
        seq_len = tl.minimum(seq_len + 1, max_model_len)
    tl.store(seq_lens_ptr + req_idx, seq_len)


@triton.jit
def _update_draft_inputs_kernel(
    output_draft_tokens_ptr,
    output_draft_tokens_stride,
    next_input_hidden_states_ptr,
    next_input_hidden_states_stride,
    input_ids_ptr,
    positions_ptr,
    sample_src_positions_ptr,
    seq_lens_ptr,
    draft_tokens_ptr,
    current_draft_step_ptr,
    hidden_states_ptr,
    hidden_states_stride,
    hidden_size,
    max_model_len,
    num_speculative_steps,
    BLOCK_SIZE: tl.constexpr,
    ADVANCE_DRAFT_POSITIONS: tl.constexpr,
):
    req_idx = tl.program_id(0)

    # Write the sampled draft token into self.draft_tokens[req_idx, step].
    draft_token = tl.load(draft_tokens_ptr + req_idx)
    step = tl.load(current_draft_step_ptr)
    tl.store(
        output_draft_tokens_ptr + req_idx * output_draft_tokens_stride + step,
        draft_token,
    )

    if step >= num_speculative_steps - 1:
        # This is the final step. Skip updating draft forward inputs.
        return

    # Advance the draft sampling key.
    sample_position = tl.load(sample_src_positions_ptr + req_idx)
    tl.store(sample_src_positions_ptr + req_idx, sample_position + 1)

    # Write the sampled draft token into the input ids tensor for the next
    # forward pass.
    tl.store(input_ids_ptr + req_idx, draft_token)

    # Copy hidden states into the input hidden states tensor for the next
    # forward pass.
    for i in range(0, hidden_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < hidden_size
        hidden_states = tl.load(
            hidden_states_ptr + req_idx * hidden_states_stride + block,
            mask=mask,
        )
        tl.store(
            next_input_hidden_states_ptr
            + req_idx * next_input_hidden_states_stride
            + block,
            hidden_states,
            mask=mask,
        )

    if ADVANCE_DRAFT_POSITIONS:
        # Increment position and seq_lens.
        # NOTE(woosuk): To prevent out-of-range access, we clamp these values
        # if they reach the max model length.
        position = tl.load(positions_ptr + req_idx)
        position = tl.minimum(position + 1, max_model_len - 1)
        tl.store(positions_ptr + req_idx, position)

        seq_len = tl.load(seq_lens_ptr + req_idx)
        seq_len = tl.minimum(seq_len + 1, max_model_len)
        tl.store(seq_lens_ptr + req_idx, seq_len)


class TritonKernels(ModelRunnerKernels):
    def prepare_prefill_inputs(
        self,
        input_ids: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        all_token_ids: torch.Tensor,
        prefill_len: torch.Tensor,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        num_reqs = idx_mapping.shape[0]
        num_lookahead = next_prefill_tokens.shape[0]
        _prepare_prefill_inputs_kernel[(num_reqs,)](
            input_ids,
            next_prefill_tokens,
            next_prefill_tokens.stride(0),
            num_lookahead,
            idx_mapping,
            query_start_loc,
            all_token_ids,
            all_token_ids.stride(0),
            prefill_len,
            num_computed_tokens,
            BLOCK_SIZE=1024,
            LOOKAHEAD_BLOCK=triton.next_power_of_2(num_lookahead),
        )

    def prepare_pos_seq_lens(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        pos: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        num_reqs = idx_mapping.shape[0]
        # NOTE(woosuk): We do +1 because the last thread block is used
        # to pad unused seq_lens as 0 for full CUDA graphs.
        _prepare_pos_seq_lens_kernel[(num_reqs + 1,)](
            pos,
            seq_lens,
            idx_mapping,
            query_start_loc,
            num_computed_tokens,
            seq_lens.shape[0],
            BLOCK_SIZE=1024,
        )

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
        num_new_sampled_tokens: int = 1,
    ) -> torch.Tensor:
        assert num_new_sampled_tokens in (0, 1), (
            f"num_new_sampled_tokens must be 0 or 1, got {num_new_sampled_tokens}"
        )
        # use idx_mapping.shape[0] for actual request count
        num_reqs = idx_mapping.shape[0]
        num_speculative_steps = draft_tokens.shape[-1]

        logits_indices = torch.empty(
            num_logits,
            dtype=torch.int64,
            device=input_ids.device,
        )
        _combine_sampled_and_draft_tokens_kernel[(num_reqs,)](
            input_ids,
            idx_mapping,
            last_sampled_tokens,
            query_start_loc,
            seq_lens,
            prefill_len,
            draft_tokens,
            draft_tokens.stride(0),
            cu_num_logits,
            logits_indices,
            NUM_NEW_SAMPLED_TOKENS=num_new_sampled_tokens,
            # NOTE(woosuk): Add num_new_sampled_tokens to ensure the block covers the
            # last sampled token in addition to all draft tokens.
            BLOCK_SIZE=triton.next_power_of_2(
                num_speculative_steps + num_new_sampled_tokens
            ),
        )
        return logits_indices

    def expand_idx_mapping(
        self,
        idx_mapping: torch.Tensor,
        total_num_logits: int,
        cu_num_logits: torch.Tensor,
        max_expand_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_reqs = idx_mapping.shape[0]
        expanded_idx_mapping = idx_mapping.new_empty(total_num_logits)
        expanded_local_pos = torch.empty(
            total_num_logits, dtype=torch.int32, device=idx_mapping.device
        )
        _expand_idx_mapping_kernel[(num_reqs,)](
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            cu_num_logits,
            BLOCK_SIZE=triton.next_power_of_2(max_expand_len),
        )
        return expanded_idx_mapping, expanded_local_pos

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
    ) -> None:
        num_reqs = idx_mapping.shape[0]
        _post_update_kernel[(num_reqs,)](
            idx_mapping,
            num_computed_tokens,
            last_sampled_tokens,
            output_bin_counts,
            output_bin_counts.stride(0) if output_bin_counts is not None else 0,
            sampled_tokens,
            sampled_tokens.stride(0),
            num_sampled,
            num_rejected,
            query_start_loc,
            all_token_ids,
            all_token_ids.stride(0),
            total_len,
            num_warps=1,
        )

    def post_update_num_computed_tokens(
        self,
        idx_mapping: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> None:
        num_reqs = idx_mapping.shape[0]
        _post_update_num_computed_tokens_kernel[(num_reqs,)](
            idx_mapping,
            num_computed_tokens,
            query_start_loc,
        )

    def apply_temperature(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
    ) -> None:
        num_tokens, vocab_size = logits.shape
        BLOCK_SIZE = 8192
        num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
        _temperature_kernel[(num_tokens, num_blocks)](
            logits,
            logits.stride(0),
            expanded_idx_mapping,
            temperature,
            vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    def apply_min_p(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        min_p: torch.Tensor,
    ) -> None:
        num_tokens, vocab_size = logits.shape
        BLOCK_SIZE = 1024
        _min_p_kernel[(num_tokens,)](
            logits,
            logits.stride(0),
            expanded_idx_mapping,
            min_p,
            vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )

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
        num_tokens, vocab_size = logits.shape
        BLOCK_SIZE = 8192
        num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
        _penalties_kernel[(num_tokens, num_blocks)](
            logits,
            logits.stride(0),
            expanded_idx_mapping,
            token_ids,
            expanded_local_pos,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            prompt_bin_mask,
            prompt_bin_mask.stride(0),
            output_bin_counts,
            output_bin_counts.stride(0),
            vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
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
        idx_long = expanded_idx_mapping.long()
        prompt_bin_mask.index_fill_(0, idx_long, 0)
        output_bin_counts.index_fill_(0, idx_long, 0)
        num_tokens = expanded_idx_mapping.shape[0]
        BLOCK_SIZE = 1024
        num_blocks = triton.cdiv(max_prefill_len, BLOCK_SIZE)
        _bincount_kernel[(num_tokens, num_blocks)](
            expanded_idx_mapping,
            all_token_ids,
            all_token_ids.stride(0),
            prompt_len,
            prefill_len,
            prompt_bin_mask,
            prompt_bin_mask.stride(0),
            output_bin_counts,
            output_bin_counts.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
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
        restore_when_all_masked: torch.Tensor,
        stop_token_ids: torch.Tensor,
        check_all_masked_rows: bool = False,
    ) -> None:
        num_tokens, vocab_size = logits.shape
        BLOCK_SIZE = triton.next_power_of_2(
            max(
                allowed_token_ids.shape[-1],
                logit_bias_token_ids.shape[-1],
                stop_token_ids.shape[-1],
            )
        )
        LOGITS_BLOCK_SIZE = 8192
        _bias_kernel[(num_tokens,)](
            logits,
            logits.stride(0),
            vocab_size,
            expanded_idx_mapping,
            num_allowed_token_ids,
            allowed_token_ids,
            allowed_token_ids.stride(0),
            num_logit_bias,
            logit_bias_token_ids,
            logit_bias_token_ids.stride(0),
            logit_bias,
            logit_bias.stride(0),
            pos,
            min_lens,
            num_stop_token_ids,
            restore_when_all_masked,
            stop_token_ids,
            stop_token_ids.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
            LOGITS_BLOCK_SIZE=LOGITS_BLOCK_SIZE,
            CHECK_ALL_MASKED_ROWS=check_all_masked_rows,
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
        num_tokens = logits.shape[0]
        _bad_words_kernel[(num_tokens, max_num_bad_words)](
            logits,
            logits.stride(0),
            expanded_idx_mapping,
            bad_word_token_ids,
            bad_word_token_ids.stride(0),
            bad_word_offsets,
            bad_word_offsets.stride(0),
            num_bad_words,
            all_token_ids,
            all_token_ids.stride(0),
            prompt_len,
            total_len,
            input_ids,
            expanded_local_pos,
        )

    def gumbel_sample(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seed: torch.Tensor,
        pos: torch.Tensor,
        apply_temperature: bool,
        is_drafting: bool,
        logits_cache: torch.Tensor | None = None,
        logits_cache_col: torch.Tensor | None = None,
        use_fp64: bool = False,
    ) -> torch.Tensor:
        return gumbel_sample(
            logits,
            expanded_idx_mapping,
            temperature,
            seed,
            pos,
            apply_temperature,
            is_drafting,
            logits_cache,
            logits_cache_col,
            use_fp64,
        )

    def compute_token_logprobs(
        self, logits: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        return compute_token_logprobs(logits, token_ids)

    def compute_token_ranks(
        self, logits: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        """One-based rank of `token_ids[row]` within its logits row."""
        batch_size, vocab_size = logits.shape
        token_ranks = torch.empty(batch_size, dtype=torch.int64, device=logits.device)
        _ranks_kernel[(batch_size,)](
            token_ranks,
            logits,
            logits.stride(0),
            token_ids,
            vocab_size,
            BLOCK_SIZE=8192,  # type: ignore
        )
        return token_ranks

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
        """Column 0 is the sampled token; the rest are a request's own logprob
        token ids when it set some, else the top-k ids. `out_valid_mask` marks the
        columns written."""
        batch_size, width = out_token_ids.shape
        _fill_logprob_token_ids_kernel[(batch_size,)](
            out_token_ids,
            out_token_ids.stride(0),
            out_valid_mask,
            out_valid_mask.stride(0),
            sampled_token_ids,
            topk_token_ids,
            topk_token_ids.stride(0),
            expanded_idx_mapping,
            num_per_req_token_ids,
            per_req_token_ids,
            per_req_token_ids.stride(0),
            NUM_TOPK=num_topk,
            PADDED_COLS=triton.next_power_of_2(width - 1),
        )

    def get_num_nans(self, logits: torch.Tensor) -> torch.Tensor:
        num_reqs, vocab_size = logits.shape
        BLOCK_SIZE = 8192
        num_nans = torch.empty(num_reqs, dtype=torch.int32, device=logits.device)
        _num_nans_kernel[(num_reqs,)](
            logits,
            logits.stride(0),
            num_nans,
            vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return num_nans

    def get_num_sampled_and_rejected(
        self,
        num_sampled: torch.Tensor,
        seq_lens: torch.Tensor,
        cu_num_logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        prefill_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_reqs = idx_mapping.shape[0]
        num_rejected = torch.empty_like(num_sampled)
        _get_num_sampled_and_rejected_kernel[(num_reqs,)](
            num_sampled,
            num_rejected,
            seq_lens,
            cu_num_logits,
            idx_mapping,
            prefill_len,
        )
        return num_sampled, num_rejected

    def get_prompt_logprobs_token_ids(
        self,
        num_tokens: int,
        query_start_loc: torch.Tensor,
        idx_mapping: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        all_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        token_ids = torch.empty(
            num_tokens, dtype=torch.int64, device=idx_mapping.device
        )
        num_reqs = idx_mapping.shape[0]
        _prompt_logprobs_token_ids_kernel[(num_reqs,)](
            token_ids,
            query_start_loc,
            idx_mapping,
            num_computed_tokens,
            all_token_ids,
            all_token_ids.stride(0),
            BLOCK_SIZE=1024,
        )
        return token_ids

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
        contexts: torch.Tensor | None = None,
        watermarking: torch.Tensor | None = None,
        watermark_key: int | None = None,
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
            contexts=contexts,
            watermarking=watermarking,
            watermark_key=watermark_key,
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

    def apply_grammar_bitmask(
        self,
        logits: torch.Tensor,
        logits_indices: torch.Tensor,
        cu_num_logits: torch.Tensor,
        bitmask: torch.Tensor,
        mask_stride: int,
    ) -> None:
        num_masks, vocab_size = bitmask.shape[0], logits.shape[-1]
        BLOCK_SIZE = 8192
        grid = (num_masks, triton.cdiv(vocab_size, BLOCK_SIZE))
        _apply_grammar_bitmask_kernel[grid](
            logits,
            logits.stride(0),
            logits_indices,
            cu_num_logits,
            bitmask,
            bitmask.stride(0),
            vocab_size,
            MASK_STRIDE=mask_stride,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    def prepare_draft_prefill_inputs(
        self,
        last_token_indices: torch.Tensor,
        current_draft_step: torch.Tensor,
        input_buffers: InputBuffers,
        input_batch: InputBatch,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        max_num_reqs: int,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        _prepare_draft_prefill_inputs_kernel[(num_reqs,)](
            last_token_indices,
            current_draft_step,
            input_buffers.input_ids,
            input_buffers.positions,
            input_buffers.query_start_loc,
            input_buffers.seq_lens,
            input_batch.input_ids,
            input_batch.positions,
            input_batch.idx_mapping,
            last_sampled,
            next_prefill_tokens,
            num_sampled,
            num_rejected,
            input_batch.query_start_loc,
            input_batch.seq_lens,
            max_num_reqs,
            BLOCK_SIZE=1024,
        )
        return last_token_indices

    def prepare_draft_decode_inputs(
        self,
        draft_tokens: torch.Tensor,
        target_seq_lens: torch.Tensor,
        num_rejected: torch.Tensor,
        input_buffers: InputBuffers,
        sample_src_positions: torch.Tensor,
        max_model_len: int,
        max_num_reqs: int,
        advance_draft_positions: bool = True,
    ) -> None:
        num_reqs = draft_tokens.shape[0]
        _prepare_draft_decode_inputs_kernel[(num_reqs + 1,)](
            draft_tokens,
            draft_tokens.stride(0),
            target_seq_lens,
            num_rejected,
            input_buffers.input_ids,
            input_buffers.positions,
            sample_src_positions,
            input_buffers.query_start_loc,
            input_buffers.seq_lens,
            max_model_len,
            max_num_reqs,
            BLOCK_SIZE=1024,
            ADVANCE_DRAFT_POSITIONS=advance_draft_positions,
        )

    def update_draft_inputs(
        self,
        draft_tokens: torch.Tensor,
        current_draft_step: torch.Tensor,
        hidden_states: torch.Tensor,
        output_draft_tokens: torch.Tensor,
        next_input_hidden_states: torch.Tensor,
        input_buffers: InputBuffers,
        sample_src_positions: torch.Tensor,
        num_reqs: int,
        max_model_len: int,
        num_speculative_steps: int,
        advance_draft_positions: bool = True,
    ) -> None:
        _, hidden_size = hidden_states.shape
        _update_draft_inputs_kernel[(num_reqs,)](
            output_draft_tokens,
            output_draft_tokens.stride(0),
            next_input_hidden_states,
            next_input_hidden_states.stride(0),
            input_buffers.input_ids,
            input_buffers.positions,
            sample_src_positions,
            input_buffers.seq_lens,
            draft_tokens,
            current_draft_step,
            hidden_states,
            hidden_states.stride(0),
            hidden_size,
            max_model_len,
            num_speculative_steps,
            BLOCK_SIZE=1024,
            ADVANCE_DRAFT_POSITIONS=advance_draft_positions,
        )

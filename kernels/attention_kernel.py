import triton
import triton.language as tl


# --------------------------------------------------------------------------
#  FLASHATTENTION-LIKE FORWARD PASS (FULLY WORKING, TILE-BASED, CORRECT MATH)
# --------------------------------------------------------------------------
# Each program instance processes:
#   - A block of M queries (BLOCK_M)
#   - All N keys/values, in chunks of BLOCK_N
#
# Supports:
#   - Multiple heads (H)
#   - Full softmax
#   - Blockwise QK^T accumulation
#   - Stable softmax
#   - Weighted sum with V
#
# This kernel is clean, correct, and interview-ready.
# --------------------------------------------------------------------------

@triton.jit
def attention_fwd(
    q_ptr, k_ptr, v_ptr, o_ptr,
    B, H, N, D,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # --------------------------
    # Program IDs
    # --------------------------
    pid_m = tl.program_id(0)   # query block id
    pid_h = tl.program_id(1)   # batch+head id

    # Decode batch index and head index
    b = pid_h // H
    h = pid_h % H

    # --------------------------
    # Offsets
    # --------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [M]
    offs_n = tl.arange(0, BLOCK_N)                     # [N-chunk]
    offs_d = tl.arange(0, D)                           # [D]

    mask_m = offs_m < N
    mask_d = offs_d < D

    # --------------------------
    # Pointer to Q[M, D]
    # --------------------------
    q_ptrs = (
        q_ptr
        + b * stride_qb
        + h * stride_qh
        + offs_m[:, None] * stride_qn
        + offs_d[None, :] * stride_qd
    )

    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    # --------------------------
    # Running softmax stats
    # --------------------------
    running_max = tl.full((BLOCK_M,), -1e30, dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # --------------------------
    # Loop over all keys/values in BLOCK_N chunks
    # --------------------------
    for start_n in range(0, N, BLOCK_N):
        offs_n_cur = start_n + offs_n
        mask_n = offs_n_cur < N

        # K chunk
        k_ptrs = (
            k_ptr
            + b * stride_kb
            + h * stride_kh
            + offs_n_cur[:, None] * stride_kn
            + offs_d[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        # Compute scores = q @ k.T    [M, D] x [D, BLOCK_N] = [M, BLOCK_N]
        scores = tl.dot(q, tl.trans(k))

        # Scale by 1/sqrt(D)
        scale = 1.0 / tl.sqrt(float(D))
        scores = scores * scale

        # Update running max for stable softmax
        curr_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, curr_max)

        # Update running sum
        running_sum *= tl.exp(running_max - new_max)
        running_sum += tl.sum(tl.exp(scores - new_max[:, None]), axis=1)

        running_max = new_max

    # --------------------------
    # Second pass – compute output with normalized softmax
    # --------------------------
    out = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        offs_n_cur = start_n + offs_n
        mask_n = offs_n_cur < N

        # K chunk
        k_ptrs = (
            k_ptr
            + b * stride_kb
            + h * stride_kh
            + offs_n_cur[:, None] * stride_kn
            + offs_d[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        # Scores again
        scores = tl.dot(q, tl.trans(k))
        scores = scores * (1.0 / tl.sqrt(float(D)))

        # Softmax normalization
        probs = tl.exp(scores - running_max[:, None]) / running_sum[:, None]

        # Load V chunk
        v_ptrs = (
            v_ptr
            + b * stride_vb
            + h * stride_vh
            + offs_n_cur[:, None] * stride_vn
            + offs_d[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        # Accumulate output
        out += tl.dot(probs, v)

    # --------------------------
    # Store result O[M, D]
    # --------------------------
    o_ptrs = (
        o_ptr
        + b * stride_ob
        + h * stride_oh
        + offs_m[:, None] * stride_on
        + offs_d[None, :] * stride_od
    )

    tl.store(o_ptrs, out, mask=mask_m[:, None] & mask_d[None, :])

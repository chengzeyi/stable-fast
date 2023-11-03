import heapq
import torch
import triton
import triton.language as tl


def conv_heuristics():
    configs = [
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 32
        },
                      num_stages=2,
                      num_warps=8),
        triton.Config({
            "BLOCK_M": 256,
            "BLOCK_N": 64,
            "BLOCK_K": 32
        },
                      num_stages=2,
                      num_warps=8),
        triton.Config({
            "BLOCK_M": 256,
            "BLOCK_N": 32,
            "BLOCK_K": 32
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "BLOCK_M": 256,
            "BLOCK_N": 32,
            "BLOCK_K": 64
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "BLOCK_M": 256,
            "BLOCK_N": 16,
            "BLOCK_K": 32
        },
                      num_stages=4,
                      num_warps=2),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 128,
            "BLOCK_K": 32
        },
                      num_stages=4,
                      num_warps=8),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "BLOCK_K": 32
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 32
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 16,
            "BLOCK_K": 32
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 128
        },
                      num_stages=3,
                      num_warps=8),
        triton.Config({
            "BLOCK_M": 256,
            "BLOCK_N": 64,
            "BLOCK_K": 128
        },
                      num_stages=3,
                      num_warps=8),
        triton.Config({
            "BLOCK_M": 256,
            "BLOCK_N": 32,
            "BLOCK_K": 128
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 128,
            "BLOCK_K": 128
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "BLOCK_K": 128
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 32,
            "BLOCK_K": 64
        },
                      num_stages=4,
                      num_warps=2),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 64
        },
                      num_stages=4,
                      num_warps=2),
        # triton.Config(
        #     {"BLOCK_M": 128, "BLOCK_N": 16, "BLOCK_K": 64}, num_stages=4, num_warps=2
        # ),
    ]
    key = [
        "BATCH",
        "IN_C",
        "IN_H",
        "IN_W",
        "KERNEL_N",
        "KERNEL_H",
        "KERNEL_W",
        "OUT_H",
        "OUT_W",
        # parameters of conv
        "stride_h",
        "stride_w",
        "padding_h",
        "padding_w",
        "dilation_h",
        "dilation_w",
        "output_padding_h",
        "output_padding_w",
        "groups",
    ]
    prune_configs_by = {
        "early_config_prune": early_config_prune,
        "perf_model": estimate_conv_time,
        "top_k": 10,
    }
    return triton.autotune(configs, key, prune_configs_by=prune_configs_by)


# unpack the given idx given the order of axis of the desired 3-dim tensor
# You could view it as the reverse of flatten the idx of 3 axis in a tensor to 1-dim idx.
# order is the order of axes in tensor, innermost dimension outward
# shape is the 3D tensor's shape
def _unpack(idx, order, shape):
    if torch.is_tensor(idx):
        _12 = torch.div(idx, shape[order[0]], rounding_mode="trunc")
        _0 = idx % shape[order[0]]
        _2 = torch.div(_12, shape[order[1]], rounding_mode="trunc")
        _1 = _12 % shape[order[1]]
    else:
        _12 = idx // shape[order[0]]
        _0 = idx % shape[order[0]]
        _2 = _12 // shape[order[1]]
        _1 = _12 % shape[order[1]]
    return _0, _1, _2


def estimate_conv_time(
    # backend, device,
    num_warps,
    num_stages,
    x,
    BATCH,
    IN_C,
    IN_H,
    IN_W,
    KERNEL_N,
    KERNEL_H,
    KERNEL_W,
    OUT_H,
    OUT_W,
    BLOCK_M,
    BLOCK_K,
    BLOCK_N,
    debug=False,
    **kwargs,
):
    """return estimated running time in ms
    = max(compute, loading) + store"""
    import triton
    import triton._C.libtriton.triton as _triton
    from triton.ops.matmul_perf_model import (
        get_dram_gbps as get_dram_gbps,
        get_tflops as get_tflops,
    )

    backend = _triton.runtime.backend.CUDA
    device = torch.cuda.current_device()
    dtype = x.dtype
    dtsize = x.element_size()

    M = BATCH * OUT_H * OUT_W
    N = KERNEL_N
    K = KERNEL_H * KERNEL_W * IN_C
    num_cta_m = triton.cdiv(M, BLOCK_M)
    num_cta_n = triton.cdiv(N, BLOCK_N)
    num_cta_k = 1
    num_ctas = num_cta_m * num_cta_n * num_cta_k

    # If the input is smaller than the block size
    M, N = max(M, BLOCK_M), max(N, BLOCK_N)

    # time to compute
    total_ops = 2 * M * N * K / (1024 * 1024 * 1024)  # GOPS
    tput = get_tflops(backend, device, num_ctas, num_warps, dtype)
    compute_ms = total_ops / tput

    # time to load data
    num_sm = _triton.runtime.num_sm(backend, device)
    active_cta_ratio = min(1, num_ctas / num_sm)
    active_cta_ratio_bw1 = min(1, num_ctas /
                               32)  # 32 active ctas are enough to saturate
    active_cta_ratio_bw2 = max(min(1, (num_ctas - 32) / (108 - 32)),
                               0)  # 32-108, remaining 5%
    dram_bw = get_dram_gbps(backend, device) * (
        active_cta_ratio_bw1 * 0.95 + active_cta_ratio_bw2 * 0.05)  # in GB/s
    l2_bw = dram_bw * 4  # rough estimation (should be 4.7 for A100?)
    # assume 80% of (following) loads are in L2 cache
    load_a_dram = M * K * dtsize * (1 + 0.2 * (num_cta_n - 1))
    load_a_l2 = M * K * dtsize * 0.8 * (num_cta_n - 1)
    load_b_dram = N * K * dtsize * (1 + 0.2 * (num_cta_m - 1))
    load_b_l2 = N * K * dtsize * 0.8 * (num_cta_m - 1)
    # total
    total_dram = (load_a_dram + load_b_dram) / (1024 * 1024)  # MB
    total_l2 = (load_a_l2 + load_b_l2) / (1024 * 1024)
    # loading time in ms
    load_ms = total_dram / dram_bw + total_l2 / l2_bw

    # estimate storing time
    store_bw = dram_bw * 0.6  # :o
    store_c_dram = M * N * dtsize / (1024 * 1024)  # MB
    store_ms = store_c_dram / store_bw

    total_time_ms = max(compute_ms, load_ms) + store_ms
    if debug:
        print(f"Total time: {total_time_ms}ms, compute time: {compute_ms}ms, "
              f"loading time: {load_ms}ms, store time: {store_ms}ms, "
              f"Activate CTAs: {active_cta_ratio*100}%")
    return total_time_ms


def early_config_prune(configs, named_args):
    from triton.runtime import driver
    from triton.compiler.compiler import get_architecture_descriptor

    device = torch.cuda.current_device()
    cc = get_architecture_descriptor(None)
    # BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages
    dtsize = named_args["x"].element_size()
    # dtype = named_args["x"].dtype

    # 1. make sure we have enough smem
    pruned_configs = []
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, BLOCK_K, num_stages = (
            kw["BLOCK_M"],
            kw["BLOCK_N"],
            kw["BLOCK_K"],
            config.num_stages,
        )
        max_shared_memory = driver.utils.get_device_properties(
            device)["max_shared_mem"]
        required_shared_memory = (BLOCK_M +
                                  BLOCK_N) * BLOCK_K * num_stages * dtsize
        if required_shared_memory <= max_shared_memory:
            pruned_configs.append(config)
    configs = pruned_configs

    # group configs by (BLOCK_M,_N,_K, num_warps)
    configs_map = {}
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = (
            kw["BLOCK_M"],
            kw["BLOCK_N"],
            kw["BLOCK_K"],
            config.num_warps,
            config.num_stages,
        )

        key = (BLOCK_M, BLOCK_N, BLOCK_K, num_warps)
        if key in configs_map:
            configs_map[key].append((config, num_stages))
        else:
            configs_map[key] = [(config, num_stages)]

    pruned_configs = []
    for k, v in configs_map.items():
        BLOCK_M, BLOCK_N, BLOCK_K, num_warps = k
        if cc >= 80:
            # compute cycles (only works for ampere GPUs)
            mmas = BLOCK_M * BLOCK_N * BLOCK_K / (16 * 8 * 16)
            mma_cycles = mmas / min(4, num_warps) * 8

            ldgsts_latency = 300  # Does this matter?
            optimal_num_stages = ldgsts_latency / mma_cycles

            # nearest stages, prefer large #stages
            nearest = heapq.nsmallest(
                2,
                v,
                key=lambda x: 10 + abs(x[1] - optimal_num_stages) if
                (x[1] - optimal_num_stages) < 0 else x[1] - optimal_num_stages,
            )

            for n in nearest:
                pruned_configs.append(n[0])
        else:  # Volta & Turing only supports num_stages <= 2
            random_config = v[0][0]
            random_config.num_stages = 2
            pruned_configs.append(random_config)
    return pruned_configs


@conv_heuristics()
@triton.jit
def _kernel_delta_x_hwc(
    x,
    w,
    bias,
    y,
    # stride of tensor
    stride_xn,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_wn,
    stride_wc,
    stride_wh,
    stride_ww,
    stride_yn,
    stride_yc,
    stride_yh,
    stride_yw,
    # pointer inc for x
    delta_xh_ptr,
    delta_xw_ptr,
    delta_xc_ptr,
    # Tensor dimensions
    BATCH,
    IN_C,
    IN_H,
    IN_W,
    KERNEL_N,
    KERNEL_H,
    KERNEL_W,
    OUT_H,
    OUT_W,
    # parameters of conv
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    output_padding_h,
    output_padding_w,
    groups,
    # Metaparameters
    ACC_TYPE: tl.constexpr,
    CONV1X1_NHWC: tl.constexpr,
    # blocks in different dimension
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    # reduction tiling parameter for matmul
    BLOCK_K: tl.constexpr,
    # Super-blocking for better L2 peformance
    GROUP_H: tl.constexpr,
    WITH_BIAS: tl.constexpr,
):
    """
    each program instance computes a [BLOCK_BATCH, BLOCK_N, BLOCK_H, BLOCK_W] block of y
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of y it should compute.
    pid_nhw = tl.program_id(0)
    pid_k = tl.program_id(1)

    # offset for output y
    off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    off_y_nhw = pid_nhw * BLOCK_M + tl.arange(0, BLOCK_M)
    off_y_n = off_y_nhw // (OUT_H * OUT_W)
    off_y_hw = off_y_nhw % (OUT_H * OUT_W)
    off_y_h = off_y_hw // OUT_W + output_padding_h
    off_y_w = off_y_hw % OUT_W + output_padding_w

    # offset for the initial ptr for x
    off_x_n = off_y_n
    off_x_h = off_y_h * stride_h - padding_h
    off_x_w = off_y_w * stride_w - padding_w
    off_x_nhw = off_x_n * stride_xn + off_x_h * stride_xh + off_x_w * stride_xw
    off_x_crs = tl.arange(0, BLOCK_K)

    CRS = IN_C * KERNEL_H * KERNEL_W
    # load inc ptr of x, upade x_ptrs
    if not CONV1X1_NHWC:
        delta_xh_ptrs = delta_xh_ptr + off_x_crs
        delta_xw_ptrs = delta_xw_ptr + off_x_crs
        delta_xc_ptrs = delta_xc_ptr + off_x_crs
        delta_xh = tl.load(delta_xh_ptrs, mask=off_x_crs < CRS, other=0)
        delta_xw = tl.load(delta_xw_ptrs, mask=off_x_crs < CRS, other=0)
        delta_xc = tl.load(delta_xc_ptrs, mask=off_x_crs < CRS, other=0)
        off_x_crs_unpacked = (delta_xh * stride_xh + delta_xw * stride_xw +
                              delta_xc * stride_xc)
        x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]
    else:
        x_ptrs = x + off_x_nhw[:, None] + off_x_crs[None, :]
        delta_xh = 0
        delta_xw = 0

    mask_x = ((off_x_n < BATCH)[:, None]
              & (off_x_crs < CRS)[None, :]
              & (off_x_h[:, None] + delta_xh[None, :] >= 0)
              & (off_x_h[:, None] + delta_xh[None, :] < IN_H)
              & (off_x_w[:, None] + delta_xw[None, :] >= 0)
              & (off_x_w[:, None] + delta_xw[None, :] < IN_W))

    # offset for the inital ptr for w
    off_w_crs = tl.arange(0, BLOCK_K)
    off_w_k = off_y_k
    w_ptrs = w + off_w_crs[:, None] + off_w_k[None, :] * stride_wn
    mask_w = (off_x_crs < CRS)[:, None] & (off_w_k < KERNEL_N)[None, :]

    # ------ load x ------
    matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)
    # ------ load w ------
    matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)

    # -----------------------------------------------------------
    # allocate accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for crs in range(0, CRS, BLOCK_K):
        # ------ matrix multiplication ------
        acc += tl.dot(matrix_x, matrix_w, out_dtype=ACC_TYPE)
        # ------ update ptrs ------
        w_ptrs += BLOCK_K
        # load inc ptr of x, upade x_ptrs
        off_x_crs = crs + BLOCK_K + tl.arange(0, BLOCK_K)
        if not CONV1X1_NHWC:
            delta_xh_ptrs += BLOCK_K
            delta_xw_ptrs += BLOCK_K
            delta_xc_ptrs += BLOCK_K
            delta_xh = tl.load(delta_xh_ptrs, mask=off_x_crs < CRS, other=0)
            delta_xw = tl.load(delta_xw_ptrs, mask=off_x_crs < CRS, other=0)
            delta_xc = tl.load(delta_xc_ptrs, mask=off_x_crs < CRS, other=0)
            off_x_crs_unpacked = (delta_xh * stride_xh + delta_xw * stride_xw +
                                  delta_xc * stride_xc)
            x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]
        else:
            x_ptrs += BLOCK_K

        mask_x = ((off_x_n < BATCH)[:, None]
                  & (off_x_crs < CRS)[None, :]
                  & (off_x_h[:, None] + delta_xh[None, :] >= 0)
                  & (off_x_h[:, None] + delta_xh[None, :] < IN_H)
                  & (off_x_w[:, None] + delta_xw[None, :] >= 0)
                  & (off_x_w[:, None] + delta_xw[None, :] < IN_W))
        mask_w = (off_x_crs < CRS)[:, None] & (off_w_k < KERNEL_N)[None, :]
        # ------ prefetch ------
        # ------ load x ------
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        # ------ load w ------
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)

    if WITH_BIAS:
        acc += tl.load(bias + off_y_k)[None, :]

    acc = acc.to(y.dtype.element_ty)

    # rematerialize -- this saves some registers
    # offset for output y
    off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    off_y_nhw = pid_nhw * BLOCK_M + tl.arange(0, BLOCK_M)
    off_y_n = off_y_nhw // (OUT_H * OUT_W)
    off_y_hw = off_y_nhw % (OUT_H * OUT_W)
    # consider output padding
    off_y_h = off_y_hw // OUT_W + output_padding_h
    off_y_w = off_y_hw % OUT_W + output_padding_w

    # y ptrs in the block of [BLOCK_M, BLOCK_N]
    y_ptrs = (y + off_y_n[:, None] * stride_yn + off_y_h[:, None] * stride_yh +
              off_y_w[:, None] * stride_yw + off_y_k[None, :] * stride_yc)

    # out-of-bounds check
    mask_y = ((off_y_n < BATCH)[:, None]
              & (off_y_h < OUT_H + output_padding_h)[:, None]
              & (off_y_w < OUT_W + output_padding_w)[:, None]
              & (off_y_k < KERNEL_N)[None, :])

    tl.store(y_ptrs, acc, mask=mask_y)

    return


@conv_heuristics()
@triton.jit
def _kernel_delta_x(
    x,
    w,
    bias,
    y,
    # stride of tensor
    stride_xn,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_wn,
    stride_wc,
    stride_wh,
    stride_ww,
    stride_yn,
    stride_yc,
    stride_yh,
    stride_yw,
    # pointer inc for x
    delta_x_ptr,
    # Tensor dimensions
    BATCH,
    IN_C,
    IN_H,
    IN_W,
    KERNEL_N,
    KERNEL_H,
    KERNEL_W,
    OUT_H,
    OUT_W,
    # parameters of conv
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    output_padding_h,
    output_padding_w,
    groups,
    # Metaparameters
    ACC_TYPE: tl.constexpr,
    CONV1X1_NHWC: tl.constexpr,
    # blocks in different dimension
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    # reduction tiling parameter for matmul
    BLOCK_K: tl.constexpr,
    # Super-blocking for better L2 peformance
    GROUP_H: tl.constexpr,
    WITH_BIAS: tl.constexpr,
):
    """
    each program instance computes a [BLOCK_BATCH, BLOCK_N, BLOCK_H, BLOCK_W] block of y
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of y it should compute.
    pid_nhw = tl.program_id(0)
    pid_k = tl.program_id(1)

    # offset for output y
    off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    off_y_nhw = pid_nhw * BLOCK_M + tl.arange(0, BLOCK_M)
    off_y_n = off_y_nhw // (OUT_H * OUT_W)
    off_y_hw = off_y_nhw % (OUT_H * OUT_W)
    off_y_h = off_y_hw // OUT_W + output_padding_h
    off_y_w = off_y_hw % OUT_W + output_padding_w

    # offset for the initial ptr for x
    off_x_n = off_y_n
    off_x_h = off_y_h * stride_h - padding_h
    off_x_w = off_y_w * stride_w - padding_w
    off_x_nhw = off_x_n * stride_xn + off_x_h * stride_xh + off_x_w * stride_xw
    off_x_crs = tl.arange(0, BLOCK_K)

    CRS = IN_C * KERNEL_H * KERNEL_W
    # load inc ptr of x, upade x_ptrs
    if not CONV1X1_NHWC:
        delta_x_ptrs = delta_x_ptr + off_x_crs
        off_x_crs_unpacked = tl.load(delta_x_ptrs, mask=off_x_crs < CRS)
        x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]
    else:
        x_ptrs = x + off_x_nhw[:, None] + off_x_crs[None, :]

    mask_x = ((off_x_n < BATCH)
              & (off_x_h >= 0)
              & (off_x_h < IN_H)
              & (off_x_w >= 0)
              & (off_x_w < IN_W))[:, None] & (off_x_crs < CRS)[None, :]

    # offset for the inital ptr for w
    off_w_crs = tl.arange(0, BLOCK_K)
    off_w_k = off_y_k
    w_ptrs = w + off_w_crs[:, None] + off_w_k[None, :] * stride_wn
    mask_w = (off_x_crs < CRS)[:, None] & (off_w_k < KERNEL_N)[None, :]

    # ------ load x ------
    matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)
    # ------ load w ------
    matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)

    # -----------------------------------------------------------
    # allocate accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for crs in range(0, CRS, BLOCK_K):
        # ------ matrix multiplication ------
        acc += tl.dot(matrix_x, matrix_w, out_dtype=ACC_TYPE)
        # ------ update ptrs ------
        w_ptrs += BLOCK_K
        # load inc ptr of x, upade x_ptrs
        if not CONV1X1_NHWC:
            delta_x_ptrs += BLOCK_K
            off_x_crs = crs + BLOCK_K + tl.arange(0, BLOCK_K)
            off_x_crs_unpacked = tl.load(delta_x_ptrs,
                                         mask=off_x_crs < CRS,
                                         other=0)
            x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]
        else:
            off_x_crs = crs + BLOCK_K + tl.arange(0, BLOCK_K)
            x_ptrs += BLOCK_K

        mask_x = ((off_x_n < BATCH)
                  & (off_x_h >= 0)
                  & (off_x_h < IN_H)
                  & (off_x_w >= 0)
                  & (off_x_w < IN_W))[:, None] & (off_x_crs < CRS)[None, :]
        mask_w = (off_x_crs < CRS)[:, None] & (off_w_k < KERNEL_N)[None, :]
        # ------ prefetch ------
        # ------ load x ------
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        # ------ load w ------
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)

    if WITH_BIAS:
        acc += tl.load(bias + off_y_k)[None, :]

    acc = acc.to(y.dtype.element_ty)

    # rematerialize -- this saves some registers
    # offset for output y
    off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    off_y_nhw = pid_nhw * BLOCK_M + tl.arange(0, BLOCK_M)
    off_y_n = off_y_nhw // (OUT_H * OUT_W)
    off_y_hw = off_y_nhw % (OUT_H * OUT_W)
    # consider output padding
    off_y_h = off_y_hw // OUT_W + output_padding_h
    off_y_w = off_y_hw % OUT_W + output_padding_w

    # y ptrs in the block of [BLOCK_M, BLOCK_N]
    y_ptrs = (y + off_y_n[:, None] * stride_yn + off_y_h[:, None] * stride_yh +
              off_y_w[:, None] * stride_yw + off_y_k[None, :] * stride_yc)

    # out-of-bounds check
    mask_y = ((off_y_n < BATCH)[:, None]
              & (off_y_h < OUT_H + output_padding_h)[:, None]
              & (off_y_w < OUT_W + output_padding_w)[:, None]
              & (off_y_k < KERNEL_N)[None, :])

    tl.store(y_ptrs, acc, mask=mask_y)

    return


class _conv:
    kernel = _kernel_delta_x_hwc

    # for the contigous order of w ptr, what"s the corresponding
    # ptr changes for x in a sliding window
    @staticmethod
    def _delta_x_ptr_hwc(
        IN_C,
        KERNEL_H,
        KERNEL_W,
        dilation_h,
        dilation_w,
        stride_wc,
        stride_wh,
        stride_ww,
        stride_xc,
        stride_xh,
        stride_xw,
        device,
    ):
        # get the order of axes in w, innermost dimension outward
        stride_w_3d = [stride_wc, stride_wh, stride_ww]
        order = sorted(range(len(stride_w_3d)), key=stride_w_3d.__getitem__)
        window_size = IN_C * KERNEL_H * KERNEL_W

        r_window = torch.arange(0, window_size, 1, device=device)
        window_unpack = _unpack(r_window, order, [IN_C, KERNEL_H, KERNEL_W])
        window_unpack_c = window_unpack[order[0]]
        window_unpack_h = window_unpack[order[1]]
        window_unpack_w = window_unpack[order[2]]
        r_dilation_h = dilation_h * window_unpack_h
        r_dilation_w = dilation_w * window_unpack_w
        r_inc = window_unpack_c
        # delta_x = (
        #     r_dilation_h * stride_xh + r_dilation_w * stride_xw + r_inc * stride_xc
        # )
        # return delta_x
        return (
            r_dilation_h,
            r_dilation_w,
            r_inc,
        )

    @staticmethod
    def _delta_x_ptr(
        IN_C,
        KERNEL_H,
        KERNEL_W,
        dilation_h,
        dilation_w,
        stride_wc,
        stride_wh,
        stride_ww,
        stride_xc,
        stride_xh,
        stride_xw,
        device,
    ):
        # get the order of axes in w, innermost dimension outward
        stride_w_3d = [stride_wc, stride_wh, stride_ww]
        order = sorted(range(len(stride_w_3d)), key=stride_w_3d.__getitem__)
        window_size = IN_C * KERNEL_H * KERNEL_W

        r_window = torch.arange(0, window_size, 1, device=device)
        window_unpack = _unpack(r_window, order, [IN_C, KERNEL_H, KERNEL_W])
        window_unpack_c = window_unpack[order[0]]
        window_unpack_h = window_unpack[order[1]]
        window_unpack_w = window_unpack[order[2]]
        r_dilation_h = dilation_h * window_unpack_h
        r_dilation_w = dilation_w * window_unpack_w
        r_inc = window_unpack_c
        delta_x = (r_dilation_h * stride_xh + r_dilation_w * stride_xw +
                   r_inc * stride_xc)
        return delta_x

    @staticmethod
    def _call(
        x,
        w,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    ):
        # Q: should we check x, w, bias dtypes?
        device = x.device
        # input shapes
        shape_x = x.shape
        shape_w = w.shape
        shape_bias = bias.shape if bias is not None else None

        # indicies for the layout
        xn, xc, xh, xw = 0, 1, 2, 3
        yn, yc, yh, yw = 0, 1, 2, 3
        wn, wc, wh, ww = 0, 1, 2, 3

        # out_channel, in_channel, kernel_height, kernel_width
        kernel_size = [shape_w[wh], shape_w[ww]]
        input_size = [shape_x[xh], shape_x[xw]]
        assert (not shape_bias or shape_bias[0] == shape_w[wn]
                ), f"bias shape did not match{shape_bias} != {shape_w[wn]}"
        in_channel = shape_w[wc] * groups

        assert shape_x[
            xc] % groups == 0, "in_channels must be divisible by groups"
        assert shape_w[
            wn] % groups == 0, "out_channels must be divisible by groups"
        assert (shape_x[xc] == in_channel
                ), f"in_channel did not match {shape_x[xc]} != {in_channel}"

        assert (len(stride) == len(padding) == len(dilation) ==
                len(output_padding) == len(kernel_size) == len(input_size))

        # output shape
        shape_y = [0] * 4
        shape_y[yn] = shape_x[xn]
        shape_y[yc] = shape_w[wn]
        shape_y[yh] = (input_size[0] + 2 * padding[0] - dilation[0] *
                       (kernel_size[0] - 1) - 1 +
                       stride[0]) // stride[0] + 2 * output_padding[0]
        shape_y[yw] = (input_size[1] + 2 * padding[1] - dilation[1] *
                       (kernel_size[1] - 1) - 1 +
                       stride[1]) // stride[1] + 2 * output_padding[1]

        BATCH = shape_x[xn]
        IN_C = shape_x[xc]
        IN_H = shape_x[xh]
        IN_W = shape_x[xw]
        KERNEL_N = shape_w[wn]
        KERNEL_H = shape_w[wh]
        KERNEL_W = shape_w[ww]
        OUT_H = shape_y[yh]
        OUT_W = shape_y[yw]

        # get strides for tensors
        stride_x = x.stride()
        stride_w = w.stride()
        with_bias = bias is not None
        if with_bias:
            bias = bias.contiguous()
        else:
            bias = None

        # output layout should be the same as x
        if stride_x[xc] < stride_x[xh] and stride_x[xc] < stride_x[xw]:
            memory_format = torch.channels_last
        else:
            memory_format = torch.contiguous_format
        # allocate output
        y = torch.empty(shape_y,
                        device=device,
                        dtype=x.dtype,
                        memory_format=memory_format)
        stride_y = y.stride()

        # allocate tmp
        # WINDOW_SIZE = KERNEL_H * KERNEL_W * IN_C
        # tmp_x = torch.empty((BATCH * OUT_H * OUT_W, WINDOW_SIZE), device=device, dtype=x.dtype)
        # tmp_w = torch.empty((WINDOW_SIZE, KERNEL_N), device=device, dtype=w.dtype)
        # accumulator types
        x_dtype = x.dtype
        if x_dtype in (
                torch.float32,
                torch.bfloat16,
        ):
            ACC_TYPE = tl.float32
        elif x_dtype in (torch.float16, ):
            ACC_TYPE = tl.float16
        elif x_dtype in (torch.float64, ):
            ACC_TYPE = tl.float64
        else:
            ACC_TYPE = tl.int32
        # ACC_TYPE = (tl.float32 if x.dtype in [
        #     torch.float16, torch.bfloat16, torch.float32
        # ] else tl.int32)
        # if stride_x[xc] == 1 and stride_x > 1 and stride_y > 1:
        CONV1X1_NHWC = False
        if stride_x[xc] == 1 and KERNEL_H == 1 and KERNEL_W == 1:
            CONV1X1_NHWC = True
        #  do we need delta x ptr for h, w, c dimension each or not
        DELTA_X_PTR_HWC = (False if
                           ((padding[0] == 0 and padding[1] == 0) or
                            (KERNEL_H == 1 and KERNEL_W == 1)) else True)
        if not CONV1X1_NHWC:
            if DELTA_X_PTR_HWC:
                delta_xh, delta_xw, delta_xc = _conv._delta_x_ptr_hwc(
                    IN_C,
                    KERNEL_H,
                    KERNEL_W,
                    dilation[0],
                    dilation[1],
                    stride_w[wc],
                    stride_w[wh],
                    stride_w[ww],
                    stride_x[xc],
                    stride_x[xh],
                    stride_x[xw],
                    device,
                )
            else:
                delta_x = _conv._delta_x_ptr(
                    IN_C,
                    KERNEL_H,
                    KERNEL_W,
                    dilation[0],
                    dilation[1],
                    stride_w[wc],
                    stride_w[wh],
                    stride_w[ww],
                    stride_x[xc],
                    stride_x[xh],
                    stride_x[xw],
                    device,
                )
        else:
            delta_x = None
            delta_xh, delta_xw, delta_xc = None, None, None

        # launch kernel, 2-dim, batch*h*w, kernel
        def grid(META):
            return (
                triton.cdiv(BATCH * OUT_H * OUT_W, META["BLOCK_M"]),
                triton.cdiv(KERNEL_N, META["BLOCK_N"]),
            )

        # conv1x1 or padding==0
        if CONV1X1_NHWC or not DELTA_X_PTR_HWC:
            _kernel_delta_x[grid](
                x,
                w,
                bias,
                y,
                # stride nchw for x,w,y tensor
                stride_x[xn],
                stride_x[xc],
                stride_x[xh],
                stride_x[xw],
                stride_w[wn],
                stride_w[wc],
                stride_w[wh],
                stride_w[ww],
                stride_y[yn],
                stride_y[yc],
                stride_y[yh],
                stride_y[yw],
                # pointer inc for x
                delta_x,
                # Tensor dimensions
                BATCH,
                IN_C,
                IN_H,
                IN_W,
                KERNEL_N,
                KERNEL_H,
                KERNEL_W,
                OUT_H,
                OUT_W,
                # conv parameters
                stride[0],
                stride[1],
                padding[0],
                padding[1],
                dilation[0],
                dilation[1],
                output_padding[0],
                output_padding[1],
                groups,
                # Metaparameters
                ACC_TYPE=ACC_TYPE,
                CONV1X1_NHWC=CONV1X1_NHWC,
                # BLOCK_M=128,
                # BLOCK_N=32,
                # BLOCK_K=32,
                GROUP_H=1,
                WITH_BIAS=with_bias,
            )
        # need to know ptr update for each dimension to check if
        # the sliding window is out of bounds
        else:
            # kernel = _kernel_delta_x_hwc
            _kernel_delta_x_hwc[grid](
                x,
                w,
                bias,
                y,
                # stride nchw for x,w,y tensor
                stride_x[xn],
                stride_x[xc],
                stride_x[xh],
                stride_x[xw],
                stride_w[wn],
                stride_w[wc],
                stride_w[wh],
                stride_w[ww],
                stride_y[yn],
                stride_y[yc],
                stride_y[yh],
                stride_y[yw],
                # pointer inc for x
                delta_xh,
                delta_xw,
                delta_xc,
                # Tensor dimensions
                BATCH,
                IN_C,
                IN_H,
                IN_W,
                KERNEL_N,
                KERNEL_H,
                KERNEL_W,
                OUT_H,
                OUT_W,
                # conv parameters
                stride[0],
                stride[1],
                padding[0],
                padding[1],
                dilation[0],
                dilation[1],
                output_padding[0],
                output_padding[1],
                groups,
                # Metaparameters
                ACC_TYPE=ACC_TYPE,
                CONV1X1_NHWC=CONV1X1_NHWC,
                # BLOCK_M=128,
                # BLOCK_N=32,
                # BLOCK_K=32,
                GROUP_H=1,
                WITH_BIAS=with_bias,
            )

        # if bias is not None:
        #     if len(bias.shape) == 1:
        #         bias = bias.reshape([1, bias.shape[0], 1, 1])
        #     y += bias
        return y

    @staticmethod
    def forward(
            x,
            w,
            bias,
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            transposed=False,
            output_padding=(0, 0),
            groups=1,
    ):
        if groups != 1:
            print(f"Do not support groups = {groups}")
            return
        if transposed:
            print("Do not support transposed")
        return _conv._call(
            x,
            w,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        )


conv_forward = _conv.forward

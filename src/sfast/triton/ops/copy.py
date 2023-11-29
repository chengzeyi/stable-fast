import torch
import triton
import triton.language as tl
from itertools import product


# Stupid: https://github.com/openai/triton/issues/1589
@eval('''triton.heuristics({
    'BLOCK_M': lambda kwargs: min(4096, triton.next_power_of_2(kwargs['size_inp_0'])),
    'BATCH_STRIDE_INP_IS_1': lambda kwargs: kwargs['batch_stride_inp'] == 1,
    'STRIDE_INP_0_IS_1': lambda kwargs: kwargs['stride_inp_0'] == 1,
    'BATCH_STRIDE_OUT_IS_1': lambda kwargs: kwargs['batch_stride_out'] == 1,
    'STRIDE_OUT_0_IS_1': lambda kwargs: kwargs['stride_out_0'] == 1,
})''')
@eval('''triton.heuristics({
    'num_warps': lambda kwargs: max(1, min(16, kwargs['BLOCK_M'] // 32)),
})''')
@triton.jit
def copy_2d_kernel(
    output_ptr,
    input_ptr,
    bs,
    size_inp_0,
    batch_stride_inp,
    stride_inp_0,
    batch_stride_out,
    stride_out_0,
    BATCH_STRIDE_INP_IS_1: tl.constexpr,
    STRIDE_INP_0_IS_1: tl.constexpr,
    BATCH_STRIDE_OUT_IS_1: tl.constexpr,
    STRIDE_OUT_0_IS_1: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_batch = tl.program_id(1)
    grid_m = tl.cdiv(size_inp_0, BLOCK_M)
    pid_m = pid
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    A = input_ptr + (1 if BATCH_STRIDE_INP_IS_1 else
                     batch_stride_inp) * pid_batch + rm * (
                         1 if STRIDE_INP_0_IS_1 else stride_inp_0)
    B = output_ptr + (1 if BATCH_STRIDE_OUT_IS_1 else
                      batch_stride_out) * pid_batch + rm * (
                          1 if STRIDE_OUT_0_IS_1 else stride_out_0)
    mask = rm < size_inp_0
    a = tl.load(A, mask=mask)
    tl.store(B, a, mask=mask)


# Stupid: https://github.com/openai/triton/issues/1589
@eval('''triton.heuristics({
    'BLOCK_M': lambda kwargs: min(64, triton.next_power_of_2(kwargs['size_inp_0'])),
    'BLOCK_N': lambda kwargs: min(64, triton.next_power_of_2(kwargs['size_inp_1'])),
    'BATCH_STRIDE_INP_IS_1': lambda kwargs: kwargs['batch_stride_inp'] == 1,
    'STRIDE_INP_0_IS_1': lambda kwargs: kwargs['stride_inp_0'] == 1,
    'STRIDE_INP_1_IS_1': lambda kwargs: kwargs['stride_inp_1'] == 1,
    'BATCH_STRIDE_OUT_IS_1': lambda kwargs: kwargs['batch_stride_out'] == 1,
    'STRIDE_OUT_0_IS_1': lambda kwargs: kwargs['stride_out_0'] == 1,
    'STRIDE_OUT_1_IS_1': lambda kwargs: kwargs['stride_out_1'] == 1,
})''')
@eval('''triton.heuristics({
    'num_warps': lambda kwargs: max(1, min(16, kwargs['BLOCK_M'] * kwargs['BLOCK_N'] // 32)),
})''')
@triton.jit
def copy_3d_kernel(
    output_ptr,
    input_ptr,
    bs,
    size_inp_0,
    size_inp_1,
    batch_stride_inp,
    stride_inp_0,
    stride_inp_1,
    batch_stride_out,
    stride_out_0,
    stride_out_1,
    BATCH_STRIDE_INP_IS_1: tl.constexpr,
    STRIDE_INP_0_IS_1: tl.constexpr,
    STRIDE_INP_1_IS_1: tl.constexpr,
    BATCH_STRIDE_OUT_IS_1: tl.constexpr,
    STRIDE_OUT_0_IS_1: tl.constexpr,
    STRIDE_OUT_1_IS_1: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_batch = tl.program_id(1)
    grid_m = tl.cdiv(size_inp_0, BLOCK_M)
    grid_n = tl.cdiv(size_inp_1, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid - pid_m * grid_n
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    A = input_ptr + (
        1 if BATCH_STRIDE_INP_IS_1 else batch_stride_inp) * pid_batch + (
            rm[:, None] *
            (1 if STRIDE_INP_0_IS_1 else stride_inp_0) + rn[None, :] *
            (1 if STRIDE_INP_1_IS_1 else stride_inp_1))
    B = output_ptr + (
        1 if BATCH_STRIDE_OUT_IS_1 else batch_stride_out) * pid_batch + (
            rm[:, None] *
            (1 if STRIDE_OUT_0_IS_1 else stride_out_0) + rn[None, :] *
            (1 if STRIDE_OUT_1_IS_1 else stride_out_1))
    mask = (rm < size_inp_0)[:, None] & (rn < size_inp_1)[None, :]
    a = tl.load(A, mask=mask)
    tl.store(B, a, mask=mask)


# Stupid: https://github.com/openai/triton/issues/1589
@eval('''triton.heuristics({
    'BLOCK_M': lambda kwargs: min(32, triton.next_power_of_2(kwargs['size_inp_0'])),
    'BLOCK_N': lambda kwargs: min(32, triton.next_power_of_2(kwargs['size_inp_1'])),
    'BLOCK_K': lambda kwargs: min(32, triton.next_power_of_2(kwargs['size_inp_2'])),
    'BATCH_STRIDE_INP_IS_1': lambda kwargs: kwargs['batch_stride_inp'] == 1,
    'STRIDE_INP_0_IS_1': lambda kwargs: kwargs['stride_inp_0'] == 1,
    'STRIDE_INP_1_IS_1': lambda kwargs: kwargs['stride_inp_1'] == 1,
    'STRIDE_INP_2_IS_1': lambda kwargs: kwargs['stride_inp_2'] == 1,
    'BATCH_STRIDE_OUT_IS_1': lambda kwargs: kwargs['batch_stride_out'] == 1,
    'STRIDE_OUT_0_IS_1': lambda kwargs: kwargs['stride_out_0'] == 1,
    'STRIDE_OUT_1_IS_1': lambda kwargs: kwargs['stride_out_1'] == 1,
    'STRIDE_OUT_2_IS_1': lambda kwargs: kwargs['stride_out_2'] == 1,
})''')
@eval('''triton.heuristics({
    'num_warps': lambda kwargs: max(1, min(16, kwargs['BLOCK_M'] * kwargs['BLOCK_N'] * kwargs['BLOCK_K'] // 32)),
})''')
@triton.jit
def copy_4d_kernel(
    output_ptr,
    input_ptr,
    bs,
    size_inp_0,
    size_inp_1,
    size_inp_2,
    batch_stride_inp,
    stride_inp_0,
    stride_inp_1,
    stride_inp_2,
    batch_stride_out,
    stride_out_0,
    stride_out_1,
    stride_out_2,
    BATCH_STRIDE_INP_IS_1: tl.constexpr,
    STRIDE_INP_0_IS_1: tl.constexpr,
    STRIDE_INP_1_IS_1: tl.constexpr,
    STRIDE_INP_2_IS_1: tl.constexpr,
    BATCH_STRIDE_OUT_IS_1: tl.constexpr,
    STRIDE_OUT_0_IS_1: tl.constexpr,
    STRIDE_OUT_1_IS_1: tl.constexpr,
    STRIDE_OUT_2_IS_1: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_batch = tl.program_id(1)
    grid_m = tl.cdiv(size_inp_0, BLOCK_M)
    grid_n = tl.cdiv(size_inp_1, BLOCK_N)
    grid_k = tl.cdiv(size_inp_2, BLOCK_K)
    pid_m = pid // (grid_n * grid_k)
    pid_nk = pid - pid_m * (grid_n * grid_k)
    pid_n = pid_nk // grid_k
    pid_k = pid_nk - pid_n * grid_k
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    A = input_ptr + (
        1 if BATCH_STRIDE_INP_IS_1 else batch_stride_inp) * pid_batch + (
            rm[:, None, None] *
            (1 if STRIDE_INP_0_IS_1 else stride_inp_0) + rn[None, :, None] *
            (1 if STRIDE_INP_1_IS_1 else stride_inp_1) + rk[None, None, :] *
            (1 if STRIDE_INP_2_IS_1 else stride_inp_2))
    B = output_ptr + (
        1 if BATCH_STRIDE_OUT_IS_1 else batch_stride_out) * pid_batch + (
            rm[:, None, None] *
            (1 if STRIDE_OUT_0_IS_1 else stride_out_0) + rn[None, :, None] *
            (1 if STRIDE_OUT_1_IS_1 else stride_out_1) + rk[None, None, :] *
            (1 if STRIDE_OUT_2_IS_1 else stride_out_2))
    mask = (rm < size_inp_0)[:, None, None] & (
        rn < size_inp_1)[None, :, None] & (rk < size_inp_2)[None, None, :]
    a = tl.load(A, mask=mask)
    tl.store(B, a, mask=mask)


def copy(dst, src):
    dst_device = dst.device
    src_device = src.device
    assert dst_device.type == 'cuda'
    assert dst_device == src_device
    dst_shape = dst.shape
    src_shape = src.shape
    assert dst_shape == src_shape

    dst_strides = dst.stride()
    src_strides = src.stride()

    ndim = dst.ndim
    if ndim in (1, 2):
        if dst.ndim == 1:
            dst = dst[None, :]
            src = src[None, :]

        bsz, sz0 = dst_shape
        bsd, sd0 = dst_strides
        bss, ss0 = src_strides

        def grid(meta):
            return (triton.cdiv(sz0, meta['BLOCK_M']), bsz)

        copy_2d_kernel[grid](
            dst,
            src,
            bsz,
            sz0,
            bss,
            ss0,
            bsd,
            sd0,
        )
    elif ndim == 3:
        bs, sz0, sz1 = dst_shape
        bsd, sd0, sd1 = dst_strides
        bss, ss0, ss1 = src_strides

        def grid(meta):
            return (triton.cdiv(sz0, meta['BLOCK_M']) *
                    triton.cdiv(sz1, meta['BLOCK_N']), bs)

        copy_3d_kernel[grid](
            dst,
            src,
            bs,
            sz0,
            sz1,
            bss,
            ss0,
            ss1,
            bsd,
            sd0,
            sd1,
        )
    elif ndim == 4:
        bs, sz0, sz1, sz2 = dst_shape
        bsd, sd0, sd1, sd2 = dst_strides
        bss, ss0, ss1, ss2 = src_strides

        def grid(meta):
            return (triton.cdiv(sz0, meta['BLOCK_M']) *
                    triton.cdiv(sz1, meta['BLOCK_N']) *
                    triton.cdiv(sz2, meta['BLOCK_K']), bs)

        copy_4d_kernel[grid](
            dst,
            src,
            bs,
            sz0,
            sz1,
            sz2,
            bss,
            ss0,
            ss1,
            ss2,
            bsd,
            sd0,
            sd1,
            sd2,
        )
    else:
        raise NotImplementedError

    return dst


if __name__ == "__main__":
    import time

    def test_transpose(x):
        print('--------------------------------')
        print('Input Shape: ', x.shape)
        print('Input Bytes: ', x.numel() * x.element_size())
        begin = time.time()
        transposed = x.transpose(-1, -2)
        out = torch.empty_like(transposed).contiguous()
        torch.cuda.synchronize()
        out = out.copy_(transposed)
        torch.cuda.synchronize()
        print(torch.allclose(out, transposed))
        end = time.time()
        print('PyTorch Time: ', end - begin)

        del out

        transposed = x.transpose(-1, -2)
        out = torch.empty_like(transposed).contiguous()
        torch.cuda.synchronize()
        begin = time.time()
        out = copy(out, transposed)
        torch.cuda.synchronize()
        print(torch.allclose(out, transposed))
        end = time.time()
        print('Triton Time: ', end - begin)

    dtypes = (torch.float, torch.half)
    sizes = (
        # (512 * 512,),
        (512, 256),
        (16 * 16 * 16, 16 * 16 * 16),
        (16, 256, 512 * 128),
        (16, 512 * 128, 256),
        (16, 128, 512, 512),
        (128, 16, 64, 512),
    )
    for i in range(2):
        print('--------------------------------')
        print('Test Case: ', i)
        for s, dtype in product(sizes, dtypes):
            x = torch.randn(s, device="cuda", dtype=dtype)
            test_transpose(x)

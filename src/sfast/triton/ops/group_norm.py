import torch
try:
    from torch._prims_common import suggest_memory_format
except ImportError:
    from sfast.utils.memory_format import suggest_memory_format
import triton
import triton.language as tl
from sfast.utils.copy_func import copy_func
from . import activation
from .utils import welford_combine

act = activation.identity


def group_norm_4d_forward_kernel(
    input_ptr,
    gamma_ptr,
    beta_ptr,
    N,
    C,
    HxW,
    groups,
    eps,
    output_ptr,
    mean_ptr,
    rstd_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    group = tl.program_id(0)
    pid_batch = tl.program_id(1)

    C_G = C // groups
    GROUP_SIZE = C_G * HxW

    offset = pid_batch * C * HxW + group * GROUP_SIZE
    X = input_ptr + offset
    Y = output_ptr + offset
    _mean = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
    _m2 = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
    _weight = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
    for off in range(0, GROUP_SIZE, BLOCK_SIZE):
        r = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + r, mask=r < GROUP_SIZE).to(tl.float32)
        m2_ = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
        weight_ = (r < GROUP_SIZE).to(tl.float32)
        _mean, _m2, _weight = welford_combine(_mean, _m2, _weight, x, m2_,
                                               weight_)
    mean, m2, weight = tl.reduce((_mean, _m2, _weight), 0, welford_combine)
    var = m2 / weight
    rstd = 1. / tl.sqrt(var + eps)
    if mean_ptr is not None:
        tl.store(mean_ptr + pid_batch * groups + group, mean)
    if rstd_ptr is not None:
        tl.store(rstd_ptr + pid_batch * groups + group, rstd)

    if gamma_ptr is None and beta_ptr is None:
        for c in range(0, C_G):
            a = rstd
            b = -a * mean
            for off in range(0, HxW, BLOCK_SIZE):
                r = off + tl.arange(0, BLOCK_SIZE)
                x = tl.load(X + c * HxW + r, mask=r < HxW).to(tl.float32)
                x = a * x + b
                x = act(x)
            tl.store(Y + c * HxW + r, x, mask=r < HxW)
    else:
        for c in range(0, C_G):
            if gamma_ptr is None:
                gamma = 1.
            else:
                gamma = tl.load(gamma_ptr + group * C_G + c).to(tl.float32)
            if beta_ptr is None:
                beta = 0.
            else:
                beta = tl.load(beta_ptr + group * C_G + c).to(tl.float32)
            a = rstd * gamma
            b = beta - a * mean
            for off in range(0, HxW, BLOCK_SIZE):
                r = off + tl.arange(0, BLOCK_SIZE)
                x = tl.load(X + c * HxW + r, mask=r < HxW).to(tl.float32)
                x = a * x + b
                x = act(x)
                tl.store(Y + c * HxW + r, x, mask=r < HxW)


def create_group_norm_4d_forward_kernel(act=activation.identity):
    kernel = group_norm_4d_forward_kernel
    kernel = copy_func(kernel,
                       globals={
                           **globals(),
                           **{
                               'act': act
                           }
                       },
                       name=f'{kernel.__name__}_{act.__name__}')
    kernel = triton.heuristics({
        'BLOCK_SIZE':
        lambda kwargs: min(4096, triton.next_power_of_2(kwargs['HxW'])),
    })(triton.heuristics({
        'num_warps':
        lambda kwargs: max(1, min(16, kwargs['HxW'] // 128)),
    })(triton.jit(kernel)))
    return kernel


# Stupid: https://github.com/openai/triton/issues/1589
@eval('''triton.heuristics({
    'ROW_SIZE':
    lambda kwargs: triton.next_power_of_2(kwargs['C'] // kwargs['groups']),
    'BLOCK_SIZE':
    lambda kwargs: max(
        1, min(triton.next_power_of_2(kwargs['HxW']),
               4096 // (triton.next_power_of_2(kwargs['C'] // kwargs['groups']))
               )),
})''')
@eval('''triton.heuristics({
    'num_warps':
    lambda kwargs: max(1, min(16, kwargs['ROW_SIZE'] * kwargs['BLOCK_SIZE'] // 128)),
})''')
@triton.jit
def group_norm_4d_channels_last_forward_collect_stats_kernel(
    input_ptr,
    N,
    C,
    HxW,
    groups,
    eps,
    mean_ptr,
    rstd_ptr,
    ROW_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    group = tl.program_id(0)
    pid_batch = tl.program_id(1)

    C_G = C // groups

    offset = pid_batch * C * HxW + group * C_G
    X = input_ptr + offset
    _mean = tl.zeros((BLOCK_SIZE, ROW_SIZE), dtype=tl.float32)
    _m2 = tl.zeros((BLOCK_SIZE, ROW_SIZE), dtype=tl.float32)
    _weight = tl.zeros((BLOCK_SIZE, ROW_SIZE), dtype=tl.float32)
    row = tl.arange(0, ROW_SIZE)
    for off in range(0, HxW, BLOCK_SIZE):
        r = off + tl.arange(0, BLOCK_SIZE)
        m2_ = tl.zeros((BLOCK_SIZE, ROW_SIZE), dtype=tl.float32)
        mask = (r < HxW)[:, None] & (row[None, :] < C_G)
        weight_ = mask.to(tl.float32)
        x = tl.load(X + (r * C)[:, None] + row[None, :],
                    mask=mask).to(tl.float32)
        _mean, _m2, _weight = welford_combine(_mean, _m2, _weight, x, m2_,
                                               weight_)
    _mean = tl.view(_mean, (BLOCK_SIZE * ROW_SIZE, ))
    _m2 = tl.view(_m2, (BLOCK_SIZE * ROW_SIZE, ))
    _weight = tl.view(_weight, (BLOCK_SIZE * ROW_SIZE, ))
    mean, m2, weight = tl.reduce((_mean, _m2, _weight), 0, welford_combine)
    var = m2 / weight
    rstd = 1. / tl.sqrt(var + eps)
    offset = pid_batch * groups + group
    tl.store(mean_ptr + offset, mean)
    tl.store(rstd_ptr + offset, rstd)


# Stupid: https://github.com/openai/triton/issues/1589
@eval('''triton.heuristics({
    'ROW_SIZE':
    lambda kwargs: triton.next_power_of_2(kwargs['C'] // kwargs['groups']),
    'BLOCK_SIZE':
    lambda kwargs: max(
        1, min(triton.next_power_of_2(kwargs['cluster_size']),
               4096 // (triton.next_power_of_2(kwargs['C'] // kwargs['groups']))
               )),
})''')
@eval('''triton.heuristics({
    'num_warps':
    lambda kwargs: max(1, min(16, kwargs['ROW_SIZE'] * kwargs['BLOCK_SIZE'] // 128)),
})''')
@triton.jit
def group_norm_4d_channels_last_forward_collect_stats_kernel_stage_1(
    input_ptr,
    N,
    C,
    HxW,
    groups,
    cluster_size,
    cluster_num,
    cluster_mean_ptr,
    cluster_m2_ptr,
    cluster_weight_ptr,
    ROW_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    group = tl.program_id(0)
    cluster = tl.program_id(1)
    pid_batch = tl.program_id(2)

    C_G = C // groups

    offset = pid_batch * C * HxW + group * C_G
    X = input_ptr + offset
    _mean = tl.zeros((BLOCK_SIZE, ROW_SIZE), dtype=tl.float32)
    _m2 = tl.zeros((BLOCK_SIZE, ROW_SIZE), dtype=tl.float32)
    _weight = tl.zeros((BLOCK_SIZE, ROW_SIZE), dtype=tl.float32)
    row = tl.arange(0, ROW_SIZE)
    start = cluster * cluster_size
    end = start + cluster_size
    end = min(end, HxW)
    for off in range(start, end, BLOCK_SIZE):
        r = off + tl.arange(0, BLOCK_SIZE)
        m2_ = tl.zeros((BLOCK_SIZE, ROW_SIZE), dtype=tl.float32)
        mask = (r < end)[:, None] & (row[None, :] < C_G)
        weight_ = mask.to(tl.float32)
        x = tl.load(X + (r * C)[:, None] + row[None, :],
                    mask=mask).to(tl.float32)
        _mean, _m2, _weight = welford_combine(_mean, _m2, _weight, x, m2_,
                                               weight_)
    _mean = tl.view(_mean, (BLOCK_SIZE * ROW_SIZE, ))
    _m2 = tl.view(_m2, (BLOCK_SIZE * ROW_SIZE, ))
    _weight = tl.view(_weight, (BLOCK_SIZE * ROW_SIZE, ))
    mean, m2, weight = tl.reduce((_mean, _m2, _weight), 0, welford_combine)
    # var = m2 / weight
    # rstd = 1. / tl.sqrt(var + eps)
    offset = pid_batch * groups * cluster_num + group * cluster_num + cluster
    tl.store(cluster_mean_ptr + offset, mean)
    tl.store(cluster_m2_ptr + offset, m2)
    tl.store(cluster_weight_ptr + offset, weight)


@eval('''triton.heuristics({
    'BLOCK_SIZE':
    lambda kwargs: triton.next_power_of_2(kwargs['cluster_num']),
})''')
@eval('''triton.heuristics({
    'num_warps':
    lambda kwargs: max(1, min(16, kwargs['BLOCK_SIZE'] // 128)),
})''')
@triton.jit
def group_norm_4d_channels_last_forward_collect_stats_kernel_stage_2(
    cluster_mean_ptr,
    cluster_m2_ptr,
    cluster_weight_ptr,
    N,
    groups,
    cluster_num,
    eps,
    mean_ptr,
    rstd_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    group = tl.program_id(0)
    pid_batch = tl.program_id(1)

    block = tl.arange(0, BLOCK_SIZE)
    mask = block < cluster_num
    offset = pid_batch * groups * cluster_num + group * cluster_num + block
    cluster_mean = tl.load(cluster_mean_ptr + offset, mask=mask)
    cluster_m2 = tl.load(cluster_m2_ptr + offset, mask=mask)
    cluster_weight = tl.load(cluster_weight_ptr + offset, mask=mask)
    mean, m2, weight = tl.reduce((cluster_mean, cluster_m2, cluster_weight), 0,
                                 welford_combine)
    var = m2 / weight
    rstd = 1. / tl.sqrt(var + eps)
    offset = pid_batch * groups + group
    tl.store(mean_ptr + offset, mean)
    tl.store(rstd_ptr + offset, rstd)


def group_norm_4d_channels_last_forward_apply_kernel(
    input_ptr,
    gamma_ptr,
    beta_ptr,
    mean_ptr,
    rstd_ptr,
    N,
    C,
    HxW,
    groups,
    eps,
    output_ptr,
    ROW_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    hw = tl.program_id(0) * BLOCK_SIZE
    pid_batch = tl.program_id(1)

    C_G = C // groups

    offset = pid_batch * C * HxW
    X = input_ptr + offset
    Y = output_ptr + offset
    group_row = tl.arange(0, ROW_SIZE)
    group_row = group_row // C_G
    group_mask = group_row < groups
    mean = tl.load(mean_ptr + pid_batch * groups + group_row, mask=group_mask)
    rstd = tl.load(rstd_ptr + pid_batch * groups + group_row, mask=group_mask)
    row = tl.arange(0, ROW_SIZE)
    mask = row < C
    if gamma_ptr is None:
        gamma = tl.full((ROW_SIZE, ), 1., dtype=mean.dtype)
    else:
        gamma = tl.load(gamma_ptr + row, mask=mask)
    if beta_ptr is None:
        beta = tl.zeros((ROW_SIZE, ), dtype=mean.dtype)
    else:
        beta = tl.load(beta_ptr + row, mask=mask)
    a = rstd * gamma
    b = beta - a * mean
    a = a[None, :]
    b = b[None, :]
    r = hw + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + (r * C)[:, None] + row[None, :],
                mask=(r < HxW)[:, None] & mask[None, :])
    x = a * x + b
    x = act(x)
    tl.store(Y + (r * C)[:, None] + row[None, :],
             x,
             mask=(r < HxW)[:, None] & mask[None, :])


def create_group_norm_4d_channels_last_forward_apply_kernel(
        act=activation.identity):
    kernel = group_norm_4d_channels_last_forward_apply_kernel
    kernel = copy_func(kernel,
                       globals={
                           **globals(),
                           **{
                               'act': act
                           }
                       },
                       name=f'{kernel.__name__}_{act.__name__}')
    kernel = triton.heuristics({
        'ROW_SIZE':
        lambda kwargs: triton.next_power_of_2(kwargs['C']),
        'BLOCK_SIZE':
        lambda kwargs: max(
            1,
            min(triton.next_power_of_2(kwargs['HxW']), 4096 // triton.
                next_power_of_2(kwargs['C']))),
    })(triton.heuristics({
        'num_warps':
        lambda kwargs: max(
            1, min(16, kwargs['ROW_SIZE'] * kwargs['BLOCK_SIZE'] // 128)),
    })(triton.jit(kernel)))
    return kernel


def create_group_norm_forward(act=activation.identity):
    group_norm_4d_forward_kernel = create_group_norm_4d_forward_kernel(act=act)
    group_norm_4d_channels_last_forward_apply_kernel = create_group_norm_4d_channels_last_forward_apply_kernel(
        act=act)

    def group_norm_forward(input,
                           num_groups,
                           weight=None,
                           bias=None,
                           eps=1e-05,
                           output_mean=True,
                           output_rstd=True):
        assert input.device.type == 'cuda'
        assert 2 <= input.ndim <= 4
        dim_padding = 0
        while input.ndim < 4:
            input = input.unsqueeze(-1)
            dim_padding += 1
        shape = input.shape
        N, C, H, W = shape
        assert C % num_groups == 0
        assert weight is None or weight.shape == (C, )
        assert bias is None or bias.shape == (C, )
        if weight is not None:
            assert weight.device.type == 'cuda'
            weight = weight.contiguous()
        if bias is not None:
            assert bias.device.type == 'cuda'
            bias = bias.contiguous()
        memory_format = suggest_memory_format(input)
        if memory_format == torch.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

            # cluster_num = 32
            # cluster_size = (H * W + cluster_num - 1) // cluster_num

            # mean_cluster = torch.empty((N, num_groups, cluster_num),
            #                            dtype=torch.float32,
            #                            device=input.device)
            # m2_cluster = torch.empty((N, num_groups, cluster_num),
            #                          dtype=torch.float32,
            #                          device=input.device)
            # weight_cluster = torch.empty((N, num_groups, cluster_num),
            #                              dtype=torch.float32,
            #                              device=input.device)

            # def grid(meta):
            #     return (num_groups, cluster_num, N)

            # group_norm_4d_channels_last_forward_collect_stats_kernel_stage_1[
            #     grid](input, N, C, H * W, num_groups, cluster_size,
            #           cluster_num, mean_cluster, m2_cluster, weight_cluster)

            mean = torch.empty((
                N,
                num_groups,
            ),
                               dtype=input.dtype,
                               device=input.device)
            rstd = torch.empty((
                N,
                num_groups,
            ),
                               dtype=input.dtype,
                               device=input.device)

            # def grid(meta):
            #     return (num_groups, N)

            # group_norm_4d_channels_last_forward_collect_stats_kernel_stage_2[
            #     grid](mean_cluster, m2_cluster, weight_cluster, N, num_groups,
            #           cluster_num, eps, mean, rstd)

            # del mean_cluster, m2_cluster, weight_cluster

            def grid(meta):
                return (num_groups, N)

            group_norm_4d_channels_last_forward_collect_stats_kernel[grid](
                input, N, C, H * W, num_groups, eps, mean, rstd)

            output = torch.empty_like(input)

            def grid(meta):
                return (triton.cdiv(H * W, meta['BLOCK_SIZE']), N)

            group_norm_4d_channels_last_forward_apply_kernel[grid](
                input, weight, bias, mean, rstd, N, C, H * W, num_groups, eps,
                output)

            if not output_mean:
                mean = None
            if not output_rstd:
                rstd = None
        else:
            mean = torch.empty(
                (
                    N,
                    num_groups,
                ), dtype=input.dtype,
                device=input.device) if output_mean else None
            rstd = torch.empty(
                (
                    N,
                    num_groups,
                ), dtype=input.dtype,
                device=input.device) if output_rstd else None

            input = input.contiguous()
            output = torch.empty_like(input)

            def grid(meta):
                return (num_groups, N)

            group_norm_4d_forward_kernel[grid](input, weight, bias, N, C,
                                               H * W, num_groups, eps, output,
                                               mean, rstd)
        while dim_padding > 0:
            output = output.squeeze(-1)
            dim_padding -= 1

        return output, mean, rstd

    return group_norm_forward


group_norm_forward = create_group_norm_forward()
group_norm_silu_forward = create_group_norm_forward(act=activation.silu)

if __name__ == '__main__':
    import time
    import torch.nn.functional as F

    def test_group_norm():
        x = torch.randn(2, 320, 32, 32).cuda().half()
        groups = 32
        weight = torch.randn(x.shape[1]).cuda().half()
        beta = torch.randn(x.shape[1]).cuda().half()

        torch.cuda.synchronize()
        begin_time = time.time()
        y = F.group_norm(x, groups, weight, beta, 1e-5)
        torch.cuda.synchronize()
        print('torch time: ', time.time() - begin_time)

        torch.cuda.synchronize()
        begin_time = time.time()
        y_triton = group_norm_forward(x, groups, weight, beta, 1e-5)[0]
        torch.cuda.synchronize()
        print('triton time: ', time.time() - begin_time)
        torch.testing.assert_close(y_triton, y, rtol=1e-2, atol=1e-2)

        x_channel_last = x.contiguous(memory_format=torch.channels_last)
        torch.cuda.synchronize()
        begin_time = time.time()
        y_channel_last = F.group_norm(x_channel_last, groups, weight, beta,
                                      1e-5)
        y_channel_last = y_channel_last.contiguous(
            memory_format=torch.channels_last)
        torch.cuda.synchronize()
        print('torch channels last time: ', time.time() - begin_time)

        torch.cuda.synchronize()
        begin_time = time.time()
        y_triton_channel_last = group_norm_forward(x_channel_last, groups,
                                                   weight, beta, 1e-5)[0]
        torch.cuda.synchronize()
        print('triton channels last time: ', time.time() - begin_time)
        torch.testing.assert_close(y_triton_channel_last,
                                   y_channel_last,
                                   rtol=1e-2,
                                   atol=1e-2)

    test_group_norm()
    test_group_norm()
    test_group_norm()

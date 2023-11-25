import torch
import packaging.version


def jit_pass_remove_contiguous(graph):
    torch._C._jit_pass_custom_pattern_based_rewrite_graph(
        '''
graph(%1, %2):
    %x : Tensor = aten::contiguous(%1, %2)
    return (%x)''', '''
graph(%1, %2):
    return (%1)''', graph)


def jit_pass_remove_dropout(graph):
    torch._C._jit_pass_custom_pattern_based_rewrite_graph(
        '''
graph(%1, %2, %3):
    %x : Tensor = aten::dropout(%1, %2, %3)
    return (%x)''', '''
graph(%1, %2, %3):
    return (%1)''', graph)


def jit_pass_optimize_gelu(graph):
    torch._C._jit_pass_custom_pattern_based_rewrite_graph(
        '''
graph(%1, %2):
    %x : Tensor = aten::gelu(%1, %2)
    return (%x)''', '''
graph(%1, %2):
    %approx: str = prim::Constant[value="tanh"]()
    %x : Tensor = aten::gelu(%1, %approx)
    return (%x)''', graph)


def jit_pass_lower_conv(graph):
    jit_pass_lower_conv1d(graph)
    jit_pass_lower_conv2d(graph)
    jit_pass_lower_conv3d(graph)


def jit_pass_optimize_cnn(graph):
    jit_pass_remove_conv_bias_followed_by_norm(graph)
    # https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBiasActivationForward
    jit_pass_fuse_conv_bias_add_sigmoid(graph)
    jit_pass_fuse_conv_bias_add_relu(graph)
    jit_pass_fuse_conv_bias_add_tanh(graph)
    jit_pass_fuse_conv_bias_sigmoid(graph)
    jit_pass_fuse_conv_bias_relu(graph)
    jit_pass_fuse_conv_bias_tanh(graph)
    jit_pass_fuse_conv_bias_add(graph)
    jit_pass_fuse_conv_bias(graph)


def jit_pass_optimize_linear(graph):
    jit_pass_fuse_linear_relu(graph)
    if torch.cuda.is_available() and packaging.version.parse(
            torch.version.cuda) >= packaging.version.parse('11.4'):
        jit_pass_fuse_linear_gelu(graph)


def jit_pass_prefer_lowp_gemm(graph):
    if hasattr(torch.ops.sfast, 'cublas_lowp_addmm'):
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%bias, %input, %weight, %beta, %alpha):
    %output : Tensor = aten::addmm(%bias, %input, %weight, %beta, %alpha)
    return (%output)''', '''
graph(%bias, %input, %weight, %beta, %alpha):
    %output : Tensor = sfast::cublas_lowp_addmm(%bias, %input, %weight, %beta, %alpha)
    return (%output)''', graph)

    if hasattr(torch.ops.sfast, 'cublas_lowp_addmm_activation'):
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%bias, %input, %weight, %beta, %alpha, %use_relu):
    %output : Tensor = aten::addmm_activation(%bias, %input, %weight, %beta, %alpha, %use_relu)
    return (%output)''', '''
graph(%bias, %input, %weight, %beta, %alpha, %use_relu):
    %output : Tensor = sfast::cublas_lowp_addmm_activation(%bias, %input, %weight, %beta, %alpha, %use_relu)
    return (%output)''', graph)

    if hasattr(torch.ops.sfast, 'cublas_lowp_mm'):
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%self, %other):
    %output : Tensor = aten::mm(%self, %other)
    return (%output)''', '''
graph(%self, %other):
    %output : Tensor = sfast::cublas_lowp_mm(%self, %other)
    return (%output)''', graph)

    if hasattr(torch.ops.sfast, 'cublas_lowp_baddbmm'):
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%bias, %input, %weight, %beta, %alpha):
    %output : Tensor = aten::baddbmm(%bias, %input, %weight, %beta, %alpha)
    return (%output)''', '''
graph(%bias, %input, %weight, %beta, %alpha):
    %output : Tensor = sfast::cublas_lowp_baddbmm(%bias, %input, %weight, %beta, %alpha)
    return (%output)''', graph)

    if hasattr(torch.ops.sfast, 'cublas_lowp_bmm'):
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%self, %other):
    %output : Tensor = aten::bmm(%self, %other)
    return (%output)''', '''
graph(%self, %other):
    %output : Tensor = sfast::cublas_lowp_bmm(%self, %other)
    return (%output)''', graph)

    if hasattr(torch.ops.sfast, 'cublas_lowp_matmul'):
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%tensor1, %tensor2):
    %output : Tensor = aten::matmul(%tensor1, %tensor2)
    return (%output)''', '''
graph(%tensor1, %tensor2):
    %output : Tensor = sfast::cublas_lowp_matmul(%tensor1, %tensor2)
    return (%output)''', graph)

    if hasattr(torch.ops.sfast, 'cublas_lowp_linear'):
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%input, %weight, %bias):
    %output : Tensor = aten::linear(%input, %weight, %bias)
    return (%output)''', '''
graph(%input, %weight, %bias):
    %output : Tensor = sfast::cublas_lowp_linear(%input, %weight, %bias)
    return (%output)''', graph)

    if hasattr(torch.ops.sfast, 'cublas_lowp_linear_relu'):
        if hasattr(torch.ops.sfast, 'linear_relu'):
            torch._C._jit_pass_custom_pattern_based_rewrite_graph(
                '''
graph(%input, %weight, %bias):
    %output : Tensor = sfast::linear_relu(%input, %weight, %bias)
    return (%output)''', '''
graph(%input, %weight, %bias):
    %output : Tensor = sfast::cublas_lowp_linear_relu(%input, %weight, %bias)
    return (%output)''', graph)

    if hasattr(torch.ops.sfast, 'cublas_lowp_linear_gelu'):
        if hasattr(torch.ops.sfast, 'linear_gelu'):
            torch._C._jit_pass_custom_pattern_based_rewrite_graph(
                '''
graph(%input, %weight, %bias):
    %output : Tensor = sfast::linear_gelu(%input, %weight, %bias)
    return (%output)''', '''
graph(%input, %weight, %bias):
    %output : Tensor = sfast::cublas_lowp_linear_gelu(%input, %weight, %bias)
    return (%output)''', graph)


def jit_pass_fuse_lowp_linear_add(graph):
    if hasattr(torch.ops.sfast, 'cublas_lowp_linear'):
        if hasattr(torch.ops.sfast, 'cublas_lowp_linear_add'):
            torch._C._jit_pass_custom_pattern_based_rewrite_graph(
                '''
graph(%input, %weight, %bias, %other, %alpha):
    %x : Tensor = sfast::cublas_lowp_linear(%input, %weight, %bias)
    %y : Tensor = aten::add(%x, %other, %alpha)
    return (%y)''', '''
graph(%input, %weight, %bias, %other, %alpha):
    %x : Tensor = sfast::cublas_lowp_linear_add(%input, %weight, %bias, %other, %alpha)
    return (%x)''', graph)

            torch._C._jit_pass_custom_pattern_based_rewrite_graph(
                '''
graph(%input, %weight, %bias, %other, %alpha):
    %x : Tensor = sfast::cublas_lowp_linear(%input, %weight, %bias)
    %y : Tensor = aten::add_(%x, %other, %alpha)
    return (%y)''', '''
graph(%input, %weight, %bias, %other, %alpha):
    %x : Tensor = sfast::cublas_lowp_linear_add(%input, %weight, %bias, %other, %alpha)
    return (%x)''', graph)

            torch._C._jit_pass_custom_pattern_based_rewrite_graph(
                '''
graph(%input, %weight, %bias, %other, %alpha):
    %x : Tensor = sfast::cublas_lowp_linear(%input, %weight, %bias)
    %y : Tensor = aten::add(%other, %x, %alpha)
    return (%y)''', '''
graph(%input, %weight, %bias, %other, %alpha):
    %x : Tensor = sfast::cublas_lowp_linear_add(%input, %weight, %bias, %other, %alpha)
    return (%x)''', graph)

            torch._C._jit_pass_custom_pattern_based_rewrite_graph(
                '''
graph(%input, %weight, %bias, %other, %alpha):
    %x : Tensor = sfast::cublas_lowp_linear(%input, %weight, %bias)
    %y : Tensor = aten::add_(%other, %x, %alpha)
    return (%y)''', '''
graph(%input, %weight, %bias, %other, %alpha):
    %x : Tensor = sfast::cublas_lowp_linear_add(%input, %weight, %bias, %other, %alpha)
    return (%x)''', graph)


def jit_pass_convert_group_norm_to_layer_norm(graph):
    torch._C._jit_pass_custom_pattern_based_rewrite_graph(
        '''
graph(%input, %num_groups, %weight, %bias, %eps, %cudnn_enabled):
    %output : Tensor = aten::group_norm(%input, %num_groups, %weight, %bias, %eps, %cudnn_enabled)
    return (%output)''', '''
graph(%input, %num_groups, %weight, %bias, %eps, %cudnn_enabled):
    %input_shape : int[] = aten::size(%input)
    %n : int, %c : int, %h : int, %w : int = prim::ListUnpack(%input_shape)
    %N : int = aten::mul(%n, %num_groups)
    %minus_one: int = prim::Constant[value=-1]()
    %input_reshape : int[] = prim::ListConstruct(%N, %minus_one)
    %input_reshaped : Tensor = aten::reshape(%input, %input_reshape)
    %M : int = aten::size(%input_reshaped, %minus_one)
    %normalized_shape : int[] = prim::ListConstruct(%M)
    %none : None = prim::Constant()
    %output : Tensor = aten::layer_norm(%input_reshaped, %normalized_shape, %none, %none, %eps, %cudnn_enabled)
    %output_reshaped : Tensor = aten::reshape(%output, %input_shape)
    %one : int = prim::Constant[value=1]()
    %new_shape : int[] = prim::ListConstruct(%one, %c, %one, %one)
    %weight_reshaped : Tensor = aten::reshape(%weight, %new_shape)
    %bias_reshaped : Tensor = aten::reshape(%bias, %new_shape)
    %output_reshaped_weighted : Tensor = aten::mul(%output_reshaped, %weight_reshaped)
    %output_reshaped_weighted_biased : Tensor = aten::add(%output_reshaped_weighted, %bias_reshaped, %one)
    return (%output_reshaped_weighted_biased)''', graph)


def jit_pass_remove_conv_bias_followed_by_norm(graph):
    torch._C._jit_pass_custom_pattern_based_rewrite_graph(
        '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21):
    %x : Tensor = aten::_convolution(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13)
    %y : Tensor = aten::batch_norm(%x, %14, %15, %16, %17, %18, %19, %20, %21)
    return (%y)''', '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21):
    %none : NoneType = prim::Constant()
    %x : Tensor = aten::_convolution(%1, %2, %none, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13)
    %y : Tensor = aten::batch_norm(%x, %14, %15, %16, %17, %18, %19, %20, %21)
    return (%y)''', graph)

    torch._C._jit_pass_custom_pattern_based_rewrite_graph(
        '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21):
    %x : Tensor = aten::_convolution(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13)
    %y : Tensor = aten::instance_norm(%x, %14, %15, %16, %17, %18, %19, %20, %21)
    return (%y)''', '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21):
    %none : NoneType = prim::Constant()
    %x : Tensor = aten::_convolution(%1, %2, %none, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13)
    %y : Tensor = aten::instance_norm(%x, %14, %15, %16, %17, %18, %19, %20, %21)
    return (%y)''', graph)


def jit_pass_replace_view_with_reshape(graph):
    torch._C._jit_pass_custom_pattern_based_rewrite_graph(
        '''
graph(%1, %2):
    %a = aten::view(%1, %2)
    return (%a)''', '''
graph(%1, %2):
    %a = aten::reshape(%1, %2)
    return (%a)''', graph)


def jit_pass_lower_conv1d(graph):
    torch._C._jit_pass_custom_pattern_based_rewrite_graph(
        '''
graph(%1, %2, %3, %4, %5, %6, %7):
    %a = aten::conv1d(%1, %2, %3, %4, %5, %6, %7)
    return (%a)''', '''
graph(%1, %2, %3, %4, %5, %6, %7):
    %false : bool = prim::Constant[value=0]()
    %true : bool = prim::Constant[value=1]()
    %zero : int = prim::Constant[value=0]()
    %output_padding : int[] = prim::ListConstruct(%zero)
    %a = aten::_convolution(%1, %2, %3, %4, %5, %6, %false, %output_padding, %7, %false, %false, %true, %true)
    return (%a)''', graph)


def jit_pass_lower_conv2d(graph):
    torch._C._jit_pass_custom_pattern_based_rewrite_graph(
        '''
graph(%1, %2, %3, %4, %5, %6, %7):
    %a = aten::conv2d(%1, %2, %3, %4, %5, %6, %7)
    return (%a)''', '''
graph(%1, %2, %3, %4, %5, %6, %7):
    %false : bool = prim::Constant[value=0]()
    %true : bool = prim::Constant[value=1]()
    %zero : int = prim::Constant[value=0]()
    %output_padding : int[] = prim::ListConstruct(%zero, %zero)
    %a = aten::_convolution(%1, %2, %3, %4, %5, %6, %false, %output_padding, %7, %false, %false, %true, %true)
    return (%a)''', graph)


def jit_pass_lower_conv3d(graph):
    torch._C._jit_pass_custom_pattern_based_rewrite_graph(
        '''
graph(%1, %2, %3, %4, %5, %6, %7):
    %a = aten::conv3d(%1, %2, %3, %4, %5, %6, %7)
    return (%a)''', '''
graph(%1, %2, %3, %4, %5, %6, %7):
    %false : bool = prim::Constant[value=0]()
    %true : bool = prim::Constant[value=1]()
    %zero : int = prim::Constant[value=0]()
    %output_padding : int[] = prim::ListConstruct(%zero, %zero, %zero)
    %a = aten::_convolution(%1, %2, %3, %4, %5, %6, %false, %output_padding, %7, %false, %false, %true, %true)
    return (%a)''', graph)


def jit_pass_fuse_conv_bias_add(graph):
    if hasattr(torch.ops.sfast, 'cudnn_convolution_bias_add'):
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15):
    %x : Tensor = aten::_convolution(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13)
    %y : Tensor = aten::add(%x, %14, %15)
    return (%y)''', '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15):
    %x : Tensor = sfast::cudnn_convolution_bias_add(%1, %2, %3, %14, %15, %4, %5, %6, %7, %8, %9)
    return (%x)''', graph)

        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15):
    %x : Tensor = aten::_convolution(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13)
    %y : Tensor = aten::add_(%x, %14, %15)
    return (%y)''', '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15):
    %x : Tensor = sfast::cudnn_convolution_bias_add(%1, %2, %3, %14, %15, %4, %5, %6, %7, %8, %9)
    return (%x)''', graph)

        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15):
    %x : Tensor = aten::_convolution(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13)
    %y : Tensor = aten::add(%14, %x, %15)
    return (%y)''', '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15):
    %x : Tensor = sfast::cudnn_convolution_bias_add(%1, %2, %3, %14, %15, %4, %5, %6, %7, %8, %9)
    return (%x)''', graph)

        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15):
    %x : Tensor = aten::_convolution(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13)
    %y : Tensor = aten::add_(%14, %x, %15)
    return (%y)''', '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15):
    %x : Tensor = sfast::cudnn_convolution_bias_add(%1, %2, %3, %14, %15, %4, %5, %6, %7, %8, %9)
    return (%x)''', graph)


def jit_pass_fuse_conv_bias(graph):
    if hasattr(torch.ops.sfast, 'cudnn_convolution_bias'):
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13):
    %x : Tensor = aten::_convolution(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13)
    return (%x)''', '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13):
    %x : Tensor = sfast::cudnn_convolution_bias(%1, %2, %3, %4, %5, %6, %7, %8, %9)
    return (%x)''', graph)


def jit_pass_fuse_conv_bias_sigmoid(graph):
    if hasattr(torch.ops.sfast, 'cudnn_convolution_bias_sigmoid'):
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13):
    %x : Tensor = aten::_convolution(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13)
    %y : Tensor = aten::sigmoid(%x)
    return (%y)''', '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13):
    %x : Tensor = sfast::cudnn_convolution_bias_sigmoid(%1, %2, %3, %4, %5, %6, %7, %8, %9)
    return (%x)''', graph)

        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13):
    %x : Tensor = aten::_convolution(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13)
    %y : Tensor = aten::sigmoid_(%x)
    return (%y)''', '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13):
    %x : Tensor = sfast::cudnn_convolution_bias_sigmoid(%1, %2, %3, %4, %5, %6, %7, %8, %9)
    return (%x)''', graph)


def jit_pass_fuse_conv_bias_relu(graph):
    if hasattr(torch.ops.sfast, 'cudnn_convolution_bias_relu'):
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13):
    %x : Tensor = aten::_convolution(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13)
    %y : Tensor = aten::relu(%x)
    return (%y)''', '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13):
    %x : Tensor = sfast::cudnn_convolution_bias_relu(%1, %2, %3, %4, %5, %6, %7, %8, %9)
    return (%x)''', graph)

        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13):
    %x : Tensor = aten::_convolution(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13)
    %y : Tensor = aten::relu_(%x)
    return (%y)''', '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13):
    %x : Tensor = sfast::cudnn_convolution_bias_relu(%1, %2, %3, %4, %5, %6, %7, %8, %9)
    return (%x)''', graph)


def jit_pass_fuse_conv_bias_tanh(graph):
    if hasattr(torch.ops.sfast, 'cudnn_convolution_bias_tanh'):
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13):
    %x : Tensor = aten::_convolution(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13)
    %y : Tensor = aten::tanh(%x)
    return (%y)''', '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13):
    %x : Tensor = sfast::cudnn_convolution_bias_tanh(%1, %2, %3, %4, %5, %6, %7, %8, %9)
    return (%x)''', graph)

        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13):
    %x : Tensor = aten::_convolution(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13)
    %y : Tensor = aten::tanh_(%x)
    return (%y)''', '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13):
    %x : Tensor = sfast::cudnn_convolution_bias_tanh(%1, %2, %3, %4, %5, %6, %7, %8, %9)
    return (%x)''', graph)


def jit_pass_fuse_conv_bias_add_sigmoid(graph):
    if hasattr(torch.ops.sfast, 'cudnn_convolution_bias_add_sigmoid'):
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15):
    %x : Tensor = aten::_convolution(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13)
    %y : Tensor = aten::add(%x, %14, %15)
    %z : Tensor = aten::sigmoid(%y)
    return (%z)''', '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15):
    %x : Tensor = sfast::cudnn_convolution_bias_add_sigmoid(%1, %2, %3, %14, %15, %4, %5, %6, %7, %8, %9)
    return (%x)''', graph)

        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15):
    %x : Tensor = aten::_convolution(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13)
    %y : Tensor = aten::add_(%x, %14, %15)
    %z : Tensor = aten::sigmoid_(%y)
    return (%z)''', '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15):
    %x : Tensor = sfast::cudnn_convolution_bias_add_sigmoid(%1, %2, %3, %14, %15, %4, %5, %6, %7, %8, %9)
    return (%x)''', graph)

        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15):
    %x : Tensor = aten::_convolution(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13)
    %y : Tensor = aten::add(%14, %x, %15)
    %z : Tensor = aten::sigmoid(%y)
    return (%z)''', '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15):
    %x : Tensor = sfast::cudnn_convolution_bias_add_sigmoid(%1, %2, %3, %14, %15, %4, %5, %6, %7, %8, %9)
    return (%x)''', graph)

        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15):
    %x : Tensor = aten::_convolution(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13)
    %y : Tensor = aten::add_(%14, %x, %15)
    %z : Tensor = aten::sigmoid_(%y)
    return (%z)''', '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15):
    %x : Tensor = sfast::cudnn_convolution_bias_add_sigmoid(%1, %2, %3, %14, %15, %4, %5, %6, %7, %8, %9)
    return (%x)''', graph)


def jit_pass_fuse_conv_bias_add_relu(graph):
    if hasattr(torch.ops.sfast, 'cudnn_convolution_bias_add_relu'):
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15):
    %x : Tensor = aten::_convolution(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13)
    %y : Tensor = aten::add(%x, %14, %15)
    %z : Tensor = aten::relu(%y)
    return (%z)''', '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15):
    %x : Tensor = sfast::cudnn_convolution_bias_add_relu(%1, %2, %3, %14, %15, %4, %5, %6, %7, %8, %9)
    return (%x)''', graph)

        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15):
    %x : Tensor = aten::_convolution(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13)
    %y : Tensor = aten::add_(%x, %14, %15)
    %z : Tensor = aten::relu_(%y)
    return (%z)''', '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15):
    %x : Tensor = sfast::cudnn_convolution_bias_add_relu(%1, %2, %3, %14, %15, %4, %5, %6, %7, %8, %9)
    return (%x)''', graph)

        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15):
    %x : Tensor = aten::_convolution(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13)
    %y : Tensor = aten::add(%14, %x, %15)
    %z : Tensor = aten::relu(%y)
    return (%z)''', '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15):
    %x : Tensor = sfast::cudnn_convolution_bias_add_relu(%1, %2, %3, %14, %15, %4, %5, %6, %7, %8, %9)
    return (%x)''', graph)

        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15):
    %x : Tensor = aten::_convolution(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13)
    %y : Tensor = aten::add_(%14, %x, %15)
    %z : Tensor = aten::relu_(%y)
    return (%z)''', '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15):
    %x : Tensor = sfast::cudnn_convolution_bias_add_relu(%1, %2, %3, %14, %15, %4, %5, %6, %7, %8, %9)
    return (%x)''', graph)


def jit_pass_fuse_conv_bias_add_tanh(graph):
    if hasattr(torch.ops.sfast, 'cudnn_convolution_bias_add_tanh'):
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15):
    %x : Tensor = aten::_convolution(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13)
    %y : Tensor = aten::add(%x, %14, %15)
    %z : Tensor = aten::tanh(%y)
    return (%z)''', '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15):
    %x : Tensor = sfast::cudnn_convolution_bias_add_tanh(%1, %2, %3, %14, %15, %4, %5, %6, %7, %8, %9)
    return (%x)''', graph)

        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15):
    %x : Tensor = aten::_convolution(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13)
    %y : Tensor = aten::add_(%x, %14, %15)
    %z : Tensor = aten::tanh_(%y)
    return (%z)''', '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15):
    %x : Tensor = sfast::cudnn_convolution_bias_add_tanh(%1, %2, %3, %14, %15, %4, %5, %6, %7, %8, %9)
    return (%x)''', graph)

        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15):
    %x : Tensor = aten::_convolution(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13)
    %y : Tensor = aten::add(%14, %x, %15)
    %z : Tensor = aten::tanh(%y)
    return (%z)''', '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15):
    %x : Tensor = sfast::cudnn_convolution_bias_add_tanh(%1, %2, %3, %14, %15, %4, %5, %6, %7, %8, %9)
    return (%x)''', graph)

        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15):
    %x : Tensor = aten::_convolution(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13)
    %y : Tensor = aten::add_(%14, %x, %15)
    %z : Tensor = aten::tanh_(%y)
    return (%z)''', '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15):
    %x : Tensor = sfast::cudnn_convolution_bias_add_tanh(%1, %2, %3, %14, %15, %4, %5, %6, %7, %8, %9)
    return (%x)''', graph)


def jit_pass_fuse_linear_relu(graph):
    torch._C._jit_pass_custom_pattern_based_rewrite_graph(
        '''
graph(%1, %2, %3):
    %x : Tensor = aten::linear(%1, %2, %3)
    %y : Tensor = aten::relu(%x)
    return (%y)''', '''
graph(%1, %2, %3):
    %y : Tensor = sfast::linear_relu(%1, %2, %3)
    return (%y)''', graph)

    torch._C._jit_pass_custom_pattern_based_rewrite_graph(
        '''
graph(%1, %2, %3):
    %x : Tensor = aten::linear(%1, %2, %3)
    %y : Tensor = aten::relu_(%x)
    return (%y)''', '''
graph(%1, %2, %3):
    %y : Tensor = sfast::linear_relu(%1, %2, %3)
    return (%y)''', graph)


def jit_pass_fuse_linear_gelu(graph):
    torch._C._jit_pass_custom_pattern_based_rewrite_graph(
        '''
graph(%1, %2, %3):
    %x : Tensor = aten::linear(%1, %2, %3)
    %y : Tensor = aten::gelu(%x)
    return (%y)''', '''
graph(%1, %2, %3):
    %y : Tensor = sfast::linear_gelu(%1, %2, %3)
    return (%y)''', graph)

    torch._C._jit_pass_custom_pattern_based_rewrite_graph(
        '''
graph(%1, %2, %3):
    %x : Tensor = aten::linear(%1, %2, %3)
    %y : Tensor = aten::gelu_(%x)
    return (%y)''', '''
graph(%1, %2, %3):
    %y : Tensor = sfast::linear_gelu(%1, %2, %3)
    return (%y)''', graph)

    torch._C._jit_pass_custom_pattern_based_rewrite_graph(
        '''
graph(%1, %2, %3, %4):
    %x : Tensor = aten::linear(%1, %2, %3)
    %y : Tensor = aten::gelu(%x, %4)
    return (%y)''', '''
graph(%1, %2, %3, %4):
    %y : Tensor = sfast::linear_gelu(%1, %2, %3)
    return (%y)''', graph)

    torch._C._jit_pass_custom_pattern_based_rewrite_graph(
        '''
graph(%1, %2, %3, %4):
    %x : Tensor = aten::linear(%1, %2, %3)
    %y : Tensor = aten::gelu_(%x, %4)
    return (%y)''', '''
graph(%1, %2, %3, %4):
    %y : Tensor = sfast::linear_gelu(%1, %2, %3)
    return (%y)''', graph)

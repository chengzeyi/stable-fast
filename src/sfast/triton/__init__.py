import logging
import sys
import functools
try:
    import triton
except ImportError:
    raise ImportError(
        'Triton is not installed. Please install it by `pip install triton`.')
import triton.language as tl
import torch

logger = logging.getLogger()


def warp_scalar_tensor_arg(func):
    '''
E               pid = tl.program_id(axis=0)
E
E               num_pid_m = tl.cdiv(M, BLOCK_M)  # number of program ids along the M axis
E                                      ^
E           IncompatibleTypeErrorImpl('invalid operands of type pointer<int64> and triton.language.int32')
    '''

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        args = (arg.item() if isinstance(arg, torch.Tensor) and arg.ndim == 0
                and arg.device.type == 'cpu' and not arg.is_floating_point()
                else arg for arg in args)
        kwargs = {
            k:
            v.item() if isinstance(v, torch.Tensor) and v.ndim == 0
            and v.device.type == 'cpu' and not v.is_floating_point() else v
            for k, v in kwargs.items()
        }
        return func(*args, **kwargs)

    return new_func


def patch_triton():
    if not hasattr(tl, 'reduce'):
        # tl.reduction is renamed to tl.reduce between 2.0.0 and 2.1.0
        tl.reduce = tl.reduction

    try:
        from triton.runtime.jit import JITFunction
    except ImportError:
        JITFunction = None

    if JITFunction is not None:
        if hasattr(JITFunction, '_make_launcher'):
            # version <= 2.1.0
            old_make_launcher = JITFunction._make_launcher

            def new_make_launcher(self, *args, **kwargs):
                launcher = old_make_launcher(self, *args, **kwargs)
                return warp_scalar_tensor_arg(launcher)

            JITFunction._make_launcher = new_make_launcher
        elif hasattr(JITFunction, 'run'):
            # version > 2.1.0
            JITFunction.run = warp_scalar_tensor_arg(JITFunction.run)
        else:
            # maybe future version
            pass

    try:
        from triton.runtime.autotuner import Autotuner
    except ImportError:
        Autotuner = None

    if Autotuner is not None:
        if hasattr(Autotuner, 'run'):
            Autotuner.run = warp_scalar_tensor_arg(Autotuner.run)
        else:
            # maybe future version
            pass

    if sys.version_info < (3, 8):
        '''
self = <triton.compiler.code_generator.CodeGenerator object at 0x7fa5a029db50>, node = <_ast.Call object at 0x7fa5a02e3a50>

    def visit_Call(self, node):
        fn = _unwrap_if_constexpr(self.visit(node.func))

        print(fn)
>       static_implementation = self.statically_implemented_functions.get(fn)
E       TypeError: unhashable type: 'tensor'

../anybus/.tox/dev/lib/python3.7/site-packages/triton/compiler/code_generator.py:938: TypeError
        '''
        '''
E           triton.compiler.errors.CompilationError: at 29:24:    C_G = C // groups
E               GROUP_SIZE = C_G * HxW
E
E               offset = pid_batch * C * HxW + group * GROUP_SIZE
E               X = input_ptr + offset
E               Y = output_ptr + offset
E               _mean = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
E               _m2 = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
E               _weight = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
E               for off in range(0, GROUP_SIZE, BLOCK_SIZE):
E                   r = off + tl.arange(0, BLOCK_SIZE)
E                   x = tl.load(X + r, mask=r < GROUP_SIZE).to(tl.float32)
E                                   ^
E           TypeError("unhashable type: 'tensor'")
        '''
        try:
            from triton.compiler.code_generator import CodeGenerator
        except ImportError:
            CodeGenerator = None

        if CodeGenerator is not None:

            class StaticallyImplementedFunctionsWrapper:

                def __init__(self, functions):
                    self.functions = functions

                def get(self, name):
                    try:
                        return self.functions.get(name)
                    except Exception:
                        return None

            CodeGenerator.statically_implemented_functions = StaticallyImplementedFunctionsWrapper(
                CodeGenerator.statically_implemented_functions)

    # try:
    #     from triton.compiler import make_launcher
    # except ImportError:
    #     make_launcher = None

    # if make_launcher is not None and hasattr(make_launcher,
    #                                          'generate_launcher'):
    #     generate_launcher = make_launcher.generate_launcher

    #     def new_generate_launcher(*args, **kwargs):
    #         src = generate_launcher(*args, **kwargs)
    #         if isinstance(src, str):
    #             # Make Triton happy with cudaMallocAsync and CUDA Graph.
    #             # Since when capturing CUDA Graph, cudaMallocAsync just creates
    #             # an 'allocate' node, does not actually allocate memory.
    #             src = src.replace('status == CUDA_ERROR_INVALID_VALUE',
    #                               'false')
    #         return src

    #     make_launcher.generate_launcher = new_generate_launcher


try:
    patch_triton()
except Exception as e:
    logger.warning(
        f'Failed to patch Trion, it might still work but not sure: {e}')

from . import torch_ops

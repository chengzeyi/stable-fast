#!/usr/bin/env python

import glob
import os
# import shutil
from os import path
from setuptools import find_packages, setup
# from typing import List
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 8], "Requires PyTorch >= 1.8"


def get_version():
    init_py_path = path.join(path.abspath(path.dirname(__file__)), "sfast",
                             "__init__.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py
                    if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")

    # The following is used to build release packages.
    # Users should never use it.
    suffix = os.getenv("SFAST_VERSION_SUFFIX", "")
    version = version + suffix
    if os.getenv("BUILD_NIGHTLY", "0") == "1":
        from datetime import datetime

        date_str = datetime.today().strftime("%y%m%d")
        version = version + ".dev" + date_str

        new_init_py = [l for l in init_py if not l.startswith("__version__")]
        new_init_py.append('__version__ = "{}"\n'.format(version))
        with open(init_py_path, "w") as f:
            f.write("".join(new_init_py))
    return version


def get_extensions():
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(this_dir, "sfast", "csrc")

    sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"),
                        recursive=True)

    from torch.utils.cpp_extension import ROCM_HOME

    is_rocm_pytorch = (True if ((torch.version.hip is not None) and
                                (ROCM_HOME is not None)) else False)
    if is_rocm_pytorch:
        assert torch_ver >= [1, 8], "ROCM support requires PyTorch >= 1.8!"

    # common code between cuda and rocm platforms, for hipify version [1,0,0] and later.
    source_cuda = glob.glob(path.join(extensions_dir, "**", "*.cu"),
                            recursive=True)

    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and
        ((CUDA_HOME is not None) or is_rocm_pytorch)) or os.getenv(
            "FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += source_cuda

        if not is_rocm_pytorch:
            define_macros += [("WITH_CUDA", None)]
            extra_compile_args["nvcc"] = [
                "-O3",
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
                "--extended-lambda",  # for fused_group_norm_impl.cu
            ]
        else:
            define_macros += [("WITH_HIP", None)]
            extra_compile_args["nvcc"] = []

        nvcc_flags_env = os.getenv("NVCC_FLAGS", "")
        if nvcc_flags_env != "":
            extra_compile_args["nvcc"].extend(nvcc_flags_env.split(" "))

        if torch_ver < [1, 7]:
            # supported by https://github.com/pytorch/pytorch/pull/43931
            CC = os.environ.get("CC", None)
            if CC is not None:
                extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "sfast._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="stable-fast",
    version=get_version(),
    author="Cheng Zeyi",
    url="https://github.com/chengzeyi/stable-fast",
    description="Stable Fast is an ultra lightweight performance optimization framework"
    " for Hugging Fase diffuser pipelines.",
    packages=find_packages(exclude=("configs", "tests*")),
    python_requires=">=3.7",
    install_requires=[
        "packaging",
        "torch>=1.12.0"
        # NOTE: When adding new dependencies, if it is required at import time (in addition
        # to runtime), it probably needs to appear in docs/requirements.txt, or as a mock
        # in docs/conf.py
    ],
    extras_require={
        # optional dependencies, required by some features
        "all": [],
        # dev dependencies. Install them by `pip install 'sfast[dev]'`
        "dev": [
            "pytest",
            "prettytable",
            "Pillow",
            "opencv-python",
            "numpy",
        ],
        "torch": [
            "torch>=1.12.0",
        ],
        "diffusers": [
            "diffusers>=0.19.0",
            "transformers",
        ],
        "xformers": [
            "xformers>=0.0.20",
        ],
        "triton": [
            "triton>=2.1.0",
        ],
    },
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)

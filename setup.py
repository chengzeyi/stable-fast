#!/usr/bin/env python

import glob
import platform
import os

# import shutil
from os import path
from setuptools import find_packages, setup

# from typing import List
import torch
from torch.utils.cpp_extension import CUDA_HOME, CUDNN_HOME, CppExtension, CUDAExtension

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 8], "Requires PyTorch >= 1.8"


def fetch_requirements():
    with open("requirements.txt") as f:
        reqs = f.read().strip().split("\n")
    return reqs


def get_version():
    if os.getenv("BUILD_VERSION"):  # In CI
        version = os.getenv("BUILD_VERSION")
    else:
        version_file_path = path.join(path.abspath(path.dirname(__file__)),
                                      "version.txt")
        version = open(version_file_path, "r").readlines()[0].strip()

    # The following is used to build release packages.
    # Users should never use it.
    suffix = os.getenv("SFAST_VERSION_SUFFIX", "")
    version = version + suffix
    if os.getenv("BUILD_NIGHTLY", "0") == "1":
        from datetime import datetime

        date_str = datetime.today().strftime("%y%m%d")
        version = version + ".dev" + date_str

    init_py_path = path.join(path.abspath(path.dirname(__file__)), "sfast",
                             "__init__.py")
    init_py = open(init_py_path, "r").readlines()
    new_init_py = [l for l in init_py if not l.startswith("__version__")]
    new_init_py.append('__version__ = "{}"\n'.format(version))
    with open(init_py_path, "w") as f:
        f.write("".join(new_init_py))
    return version


def get_extensions():
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(this_dir, "sfast", "csrc")
    include_dirs = [extensions_dir]

    sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"),
                        recursive=True)
    # common code between cuda and rocm platforms, for hipify version [1,0,0] and later.
    source_cuda = glob.glob(path.join(extensions_dir, "**", "*.cu"),
                            recursive=True)
    source_cuda_rt = glob.glob(path.join(extensions_dir, "**", "*.cc"),
                               recursive=True)

    extension = CppExtension

    extra_compile_args = {"cxx": []}
    library_dirs = []
    libraries = []
    define_macros = []

    # if (torch.cuda.is_available()
    #         and ((CUDA_HOME is not None) or is_rocm_pytorch)):
    # Skip the above useless check as we will always compile with CUDA support,
    # and the CI might be running on CPU-only machines.
    if os.getenv("WITHOUT_CUDA", "0") != "1":
        assert CUDA_HOME is not None, "Cannot find CUDA installation."

        cudnn_front_end_root = os.path.join(this_dir, "third_party", "cudnn-frontend")
        if not os.path.exists(cudnn_front_end_root):
            raise RuntimeError("Cannot find cudnn-frontend. Please run "
                                 "`git submodule update --init --recursive`.")
        cudnn_front_end_include = os.path.join(cudnn_front_end_root, "include")
        include_dirs.append(cudnn_front_end_include)

        extension = CUDAExtension
        sources += source_cuda
        sources += source_cuda_rt

        # from torch.utils.cpp_extension import ROCM_HOME

        # is_rocm_pytorch = (True if ((torch.version.hip is not None) and
        #                             (ROCM_HOME is not None)) else False)
        # if is_rocm_pytorch:
        #     assert torch_ver >= [1, 8], "ROCM support requires PyTorch >= 1.8!"

        # if not is_rocm_pytorch:
        if True:
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

        if CUDNN_HOME is None:
            try:
                # Try to use the bundled version of CUDNN with PyTorch installation.
                # This is also used in CI.
                # CUBLAS is not needed as the downloaded CUDA should include it.
                from nvidia import cudnn
            except ImportError:
                cudnn = None

            if cudnn is not None:
                print("Using CUDNN from {}".format(cudnn.__file__))
                cudnn_dir = os.path.dirname(cudnn.__file__)
                include_dirs.append(os.path.join(cudnn_dir, "include"))
                # Hope PyTorch knows how to link it correctly.
                # We only need headers because PyTorch should have
                # linked the actual library file. (But why not work on Windows?)

                # Make Windows CI happy (unresolved external symbol)
                # Why Linux does not need this?
                if platform.system() == "Windows":
                    library_dirs.append(os.path.join(cudnn_dir, "lib", "x64"))

        # Make Windows CI happy (unresolved external symbol)
        # Why Linux does not need this?
        if platform.system() == "Windows":
            libraries.append("cudnn")
            libraries.append("cublas")
    else:
        print("Compiling without CUDA support")

    ext_modules = [
        extension(
            "sfast._C",
            sorted(sources),
            include_dirs=[os.path.abspath(p) for p in include_dirs],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            library_dirs=[os.path.abspath(p) for p in library_dirs],
            libraries=libraries,
        )
    ]

    return ext_modules


setup(
    name="stable-fast",
    version=get_version(),
    author="Cheng Zeyi",
    url="https://github.com/chengzeyi/stable-fast",
    description=
    "Stable Fast is an ultra lightweight performance optimization framework"
    " for Hugging Fase diffuser pipelines.",
    packages=find_packages(exclude=("configs", "tests*")),
    python_requires=">=3.7",
    install_requires=fetch_requirements(),
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

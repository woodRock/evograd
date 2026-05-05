"""
Optional CUDA extension build.  All package metadata lives in pyproject.toml.

CPU / MPS install (no build step required):
    pip install -e .

With custom CUDA kernels (requires nvcc ≥ 11.8):
    pip install -e ".[cuda]"
    # or build the .so in-place:
    python setup.py build_ext --inplace
"""
from setuptools import setup

try:
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension

    setup(
        ext_modules=[
            CUDAExtension(
                name="evograd_cuda",
                sources=["evograd/cuda/kernels.cu"],
                extra_compile_args={
                    "cxx":  ["-O3"],
                    "nvcc": ["-O3", "--use_fast_math",
                             "-gencode", "arch=compute_80,code=sm_80"],
                },
            )
        ],
        cmdclass={"build_ext": BuildExtension},
    )
except Exception:
    setup()

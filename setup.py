"""
Build script for EvoGrad.

CPU-only install:
  pip install torch numpy && pip install -e .

With CUDA kernels (requires nvcc):
  python setup.py build_ext --inplace
"""
from setuptools import setup, find_packages

try:
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension
    ext_modules = [
        CUDAExtension(
            name="evograd_cuda",
            sources=["evograd/cuda/kernels.cu"],
            extra_compile_args={
                "cxx":  ["-O3"],
                "nvcc": ["-O3", "--use_fast_math",
                         "-gencode", "arch=compute_80,code=sm_80"],
            },
        )
    ]
    cmdclass = {"build_ext": BuildExtension}
except Exception:
    ext_modules = []
    cmdclass = {}

setup(
    name="evograd",
    version="0.1.0",
    description=(
        "Functional deep learning + GPU-accelerated evolutionary "
        "computation — one language, two paradigms."
    ),
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["torch>=2.0", "numpy"],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)

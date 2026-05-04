"""
EvoGrad — A unified language for functional deep learning and
GPU-accelerated evolutionary computation.

  from evograd import Tensor, Population, EvoEngine
  from evograd.nn import Linear, ReLU, Softmax
  from evograd.evo import rastrigin
  from evograd.compiler import parse, type_check, compile_program
  from evograd.dsl import HybridTrainer
"""
from .core import Tensor, TensorType, Float32, ShapeError
from .evo import Population, EvoEngine, EvoConfig
from .dsl import HybridTrainer, HybridConfig
from .mps.device import auto_device, sync as mps_sync, warmup_mps

__version__ = "0.1.0"
__all__ = [
    "Tensor", "TensorType", "Float32", "ShapeError",
    "Population", "EvoEngine", "EvoConfig",
    "HybridTrainer", "HybridConfig",
    "auto_device", "mps_sync", "warmup_mps",
]

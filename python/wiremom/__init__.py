"""wiremom: an open-source thin-wire method-of-moments antenna solver (MPIE formulation)."""

from .geometry import Structure
from .solver import C0, Model, Solution

__all__ = ["Structure", "Model", "Solution", "C0"]
__version__ = "0.1.0"

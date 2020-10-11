from .ISolver import ISolver
from .. import get_submodules_from_kwargs  # needs to be able throw request from down-leveled modules
from .optimizers import AccumOptimizer  # needs to simplify access
"""
from .ProdSolver_MP import ProdSolver_MP
from .ProdSolverRocks import ProdSolverRocks
from .SegmSolver import SegmSolver
from .RegrSolver import RegrSolver
from .RegrRocksSolver import RegrRocksSolver
"""


def get_native_unet(**kwargs):
    from .NativeUnet import get_solver
    return get_solver(**kwargs)


def get_segm_solver(**kwargs):
    from .SegmSolver import get_solver
    return get_solver(**kwargs)


def get_regr_solver(**kwargs):
    from .RegrSolver import get_solver
    return get_solver(**kwargs)


def get_regrrocks_solver(**kwargs):
    from .RegrRocksSolver import get_solver
    return get_solver(**kwargs)

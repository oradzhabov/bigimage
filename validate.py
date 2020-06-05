from kmodel import validate
from config import cfg
from solvers import SegmSolver

if __name__ == "__main__":
    solver = SegmSolver()
    validate.run(cfg, solver)

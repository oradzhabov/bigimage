from kmodel import train
from config import cfg
from solvers import SegmSolver

if __name__ == "__main__":
    solver = SegmSolver()
    train.run(cfg, solver)

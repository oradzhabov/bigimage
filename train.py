from kmodel import train
from config import cfg
from solvers import SegmSolver

if __name__ == "__main__":
    solver = SegmSolver(cfg)
    train.run(cfg, solver, review_augmented_sample=False)

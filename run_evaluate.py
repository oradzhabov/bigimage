from kmodel import evaluate
from config import cfg
from solvers import SegmSolver

if __name__ == "__main__":
    show_random_items_nb = 20

    solver = SegmSolver(cfg)
    evaluate.run(cfg, solver, show_random_items_nb)

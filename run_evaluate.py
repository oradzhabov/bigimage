from kmodel import evaluate
from config import cfg
from solvers import *
from data_provider import *

if __name__ == "__main__":
    show_random_items_nb = 20

    solver = SegmSolver(cfg)
    provider = SemanticSegmentationDataProvider

    evaluate.run(cfg, solver, provider, show_random_items_nb)

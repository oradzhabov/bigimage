from kmodel import evaluate
from config import cfg
from solvers import *
from data_provider import *

if __name__ == "__main__":
    show_random_items_nb = 10

    use_regression = False
    if use_regression:
        solver = RegrSolver(cfg)
        provider = RegressionSegmentationDataProvider
    else:
        solver = SegmSolver(cfg)
        provider = SemanticSegmentationDataProvider

    evaluate.run(cfg, solver, provider, show_random_items_nb)

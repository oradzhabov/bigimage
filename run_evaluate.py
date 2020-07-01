from kmodel import evaluate
from config_rocks import cfg
from solvers import *
from data_provider import *

if __name__ == "__main__":
    show_random_items_nb = 20

    use_regression = True
    if use_regression:
        solver = RegrSolver(cfg)
        provider = RegressionSegmentationDataProvider
    else:
        solver = SegmSolver(cfg)
        provider = SemanticSegmentationDataProvider

    evaluate.run(cfg, solver, provider, show_random_items_nb)

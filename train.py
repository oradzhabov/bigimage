from kmodel import train
from config import cfg
from solvers import *
from data_provider import *


if __name__ == "__main__":
    use_regression = False
    if use_regression:
        solver = RegrSolver(cfg)
        provider = RegressionSegmentationDataProvider
    else:
        solver = SegmSolver(cfg)
        provider = SemanticSegmentationDataProvider

    train.run(cfg, solver, provider, review_augmented_sample=False)

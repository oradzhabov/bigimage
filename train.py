from kmodel import train
from config import cfg
from solvers import *
from data_provider import *


if __name__ == "__main__":
    solver = SegmSolver(cfg)
    provider = SemanticSegmentationDataProvider

    train.run(cfg, solver, provider, review_augmented_sample=False)

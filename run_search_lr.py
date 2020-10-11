from .config_stockpile import cfg
from .kmodel import find_lr


def run_searching_learning_rate():
    solver = cfg.solver(cfg)
    provider = cfg.provider
    aug = cfg.aug()

    # Store initialized params to config for proper serialization with evaluation results
    cfg.solver = solver
    cfg.provider = provider
    cfg.aug = aug

    find_lr.run(cfg, solver, provider, aug, start_lr=0.00001, end_lr=1, no_epochs=20, moving_average=20)

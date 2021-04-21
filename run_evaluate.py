from .config_stockpile import cfg
from .kmodel import evaluate


def run_evaluate():
    # Suppress AccumGradOptimizer which used when "batch_size_multiplier" > 1 and does not supported by TF2.
    # Anyway it has no affect to evaluation result.
    cfg.batch_size_multiplier = 1
    #
    solver = cfg.solver(cfg)
    provider = cfg.provider
    aug = cfg.aug()

    # Store initialized params to config for proper serialization with evaluation results
    cfg.solver = solver
    cfg.provider = provider
    cfg.aug = aug

    evaluate.run(cfg, solver, provider, aug, show_random_items_nb=10, save_imgs=False)

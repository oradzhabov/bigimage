from config_stockpile import cfg
from kmodel import evaluate


if __name__ == "__main__":
    solver = cfg.solver(cfg)
    provider = cfg.provider
    aug = cfg.aug()

    # Store initialized params to config for proper serialization with evaluation results
    cfg.solver = solver
    cfg.provider = provider
    cfg.aug = aug

    evaluate.run(cfg, solver, provider, aug, show_random_items_nb=0)

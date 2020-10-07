from .config_stockpile import cfg
from .kmodel import train


if __name__ == "__main__":
    solver = cfg.solver(cfg)
    provider = cfg.provider
    aug = cfg.aug()

    # Store initialized params to config for proper serialization with evaluation results
    cfg.solver = solver
    cfg.provider = provider
    cfg.aug = aug

    train.run(cfg, solver, provider, aug, review_augmented_sample=False)

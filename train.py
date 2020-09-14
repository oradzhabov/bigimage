from config_stockpile import cfg
from kmodel import train


if __name__ == "__main__":
    solver = cfg.solver(cfg)
    provider = cfg.provider
    aug = cfg.aug()

    train.run(cfg, solver, provider, aug, review_augmented_sample=False)

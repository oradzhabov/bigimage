from config_stockpile import cfg
from kmodel import evaluate


if __name__ == "__main__":
    solver = cfg.solver(cfg)
    provider = cfg.provider
    aug = cfg.aug()

    evaluate.run(cfg, solver, provider, aug, show_random_items_nb=0)

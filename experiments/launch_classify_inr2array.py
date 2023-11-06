import hydra


@hydra.main(config_name="main", config_path="./classify_latent_configs", version_base=None)
def main(cfg=None):
    from omegaconf import OmegaConf
    from experiments.classify_inr2array import main
    print(OmegaConf.to_yaml(cfg, resolve=True))
    main(cfg)


if __name__ == "__main__":
    main()
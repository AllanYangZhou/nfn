import hydra


@hydra.main(config_path="./classify_siren_configs", config_name="main", version_base="1.1")
def main(cfg):
    from omegaconf import OmegaConf
    from experiments.classify_siren import main
    # pretty print hydra cfg
    print(OmegaConf.to_yaml(cfg, resolve=True))
    main(cfg)


if __name__ == "__main__":
    main()
import hydra


@hydra.main(config_path="./stylize_configs", config_name="main", version_base="1.1")
def main(cfg):
    from omegaconf import OmegaConf
    from experiments.stylize_siren import main
    print(OmegaConf.to_yaml(cfg, resolve=True))
    main(cfg)


if __name__ == "__main__":
    main()
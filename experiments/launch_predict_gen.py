import hydra

@hydra.main(config_path="./predict_gen_configs", config_name="main", version_base="1.1")
def main(cfg):
    from omegaconf import OmegaConf
    from experiments.predict_gen import train
    # pretty print hydra cfg
    print(OmegaConf.to_yaml(cfg, resolve=True))
    train(cfg)


if __name__ == "__main__":
    main()
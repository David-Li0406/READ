import hydra
from omegaconf import DictConfig
from hydra import compose, initialize
import wandb
import argparse

def main():
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src.stage_3.sentence_level.main import train_stage_3_sentence_re
    from src.stage_1 import utils
    wandb.init(project="erica_wat_wiki80_1")
    parser = argparse.ArgumentParser()
    parser.add_argument("--adv_lr", type=float)
    parser.add_argument("--adv_max_norm", type=float)
    parser.add_argument("--adv_steps", type=int)
    # parser.add_argument("--clean_token_leaving_prob", type=float)
    args = parser.parse_args()

    initialize(config_path="configs/")
    config = compose(config_name="config.yaml", overrides=["experiment=stage_3_sentence_re.yaml"])
    config.trainer.adv_lr = args.adv_lr
    config.trainer.adv_max_norm = args.adv_max_norm
    config.trainer.adv_steps = args.adv_steps
    # config.trainer.clean_token_leaving_prob = args.clean_token_leaving_prob
    
    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    test_score, dev_score = train_stage_3_sentence_re(config)

    wandb.log({
        'test_score': test_score,
        'dev_score': dev_score,
    })




if __name__ == "__main__":
    main()

import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src.stage_1.main import train_stage_1
    from src.stage_1.eval import eval_stage_1
    from src.stage_1 import utils
    from src.stage_2.main import train_stage_2
    from src.stage_3.sentence_level.main import train_stage_3_sentence_re
    from preprocess.preprocess_docred import preprocess_data
    from preprocess.preprocess_erica import preprocess_erica

    # Preprocess data mode
    if config.preprocess_mode:
        if config.erica_pretrain_data:
            return preprocess_erica(config)
        else:
            return preprocess_data(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    # Train
    if 'stage_1' in config.name:
        return train_stage_1(config)
    elif 'stage_2' in config.name:
        return train_stage_2(config)
    elif 'stage_3_sentence_re' in config.name:
        return train_stage_3_sentence_re(config)
    elif 'stage_3_doc_re' in config.name:
        return train_stage_1(config)

    # Eval
    if config.name == 'stage_3_eval':
        return eval_stage_1(config)


if __name__ == "__main__":
    main()
    print('Script finished.')

import os

from src.stage_1.main import Controller
from src.stage_1.utils import target_directory


def eval_stage_1(config):
    con = Controller(config)
    model_type = 'bert' if '_bert' in config.model_path_datetime else 'roberta'
    model_name_or_path = 'bert-base-uncased' if '_bert' in config.model_path_datetime else 'roberta-base'
    target_dir, best_model_path = target_directory(config.proj_root_dir, config.model_path_datetime,
                                                   model_type=model_type)

    # WCL method
    if config.wcl_method:
        con.run_wcl(config.model_type, config.model_name_or_path, best_model_path)

    # Run error analysis on dev set
    elif config.error_analysis:
        split = 'dev'
        print('-' * 89)
        print(f'Beginning evaluation of {split} split.')

        # The following assertions are required to ensure we save the learned instance from the dev set
        assert config.training_data_type == 'distant'

        con.test_prefix = split
        con.dir_first_learned_train = con.dir_first_learned_train.replace('train_distant_fl',
                                                                          'dev_final_correct_uids')  # rename dir to collect dev correct uids
        os.makedirs(con.dir_first_learned_train, exist_ok=True)
        con.test(model_type, model_name_or_path, config.trainer.save_name,
                 config.input_theta, best_model_path, target_dir)
        print('-' * 89)

    # Eval test split
    else:
        assert not config.error_analysis  # cannot do error analysis on test set
        split = 'test'
        print('-' * 89)
        print(f'Beginning evaluation of {split} split.')
        con.test_prefix = split
        con.test(model_type, model_name_or_path, config.trainer.save_name,
                 config.input_theta, best_model_path, target_dir)
        print('-' * 89)

    return

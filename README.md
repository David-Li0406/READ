<div align="center">

# READ: Improving Relation Extraction from an ADversarial Perspective

</div>

### Description

This repo contains the source code for the NAACL 2024 findings paper [READ: Improving Relation Extraction from an ADversarial Perspective](https://arxiv.org/pdf/2404.02931.pdf). 

### Project Structure

```
├── configs                <- Configuration files
│   ├── experiment               <- Experiment configs
│   ├── mode                     <- Mode configs
│   ├── trainer                  <- Trainer configs
│   ├── config.yaml              <- Main config 
|   └── wandb.yaml               <- Adversarial hyper-parameter config 
│
├── data                   <- Project data
│
├── logs                   <- Logs and saved checkpoints
│
├── preprocess             <- Preprocessing scripts
│
├── saved_models           <- Saved models
│
├── src                    <- Source code
│   ├── stage_1                  <- Stage 1 code: record learning order
│   ├── stage_2                  <- Stage 2 code: contrastive pre-training
│   └── stage_3                  <- Stage 2 code: fine-tuning
│       └── sentence_level       <- Fine-tune for sentence-level relation extraction
│
├── requirements.txt       <- File for installing python dependencies
├── run.py                 <- Controller
├── run_wandb_v2.py        <- Entry for adversarial hyper-parameter searching
└── README.md
```

### Initalize
- Install dependencies from `requirements.txt`
- Install [Apex](https://github.com/NVIDIA/apex)
- For ERICA models, you can find them [here](https://drive.google.com/drive/folders/19SxYoDeKZg4Ho_FIrDYpcifCtpsl5u3K).
- For FineCL models and dataset, you can find them [here](https://drive.google.com/drive/folders/13-iTHhde8B5BQPNk8bCA0z6dxxo42ov1?usp=sharing). Unzip `data.zip` and then move both `data` and `saved_models` into the project's root directory.

### Run
- Revise the `pretrained_model_path` parameter in `configs/experiment/stage_3_sentence_re.yaml` (FineCL or ERICA).
- Revise the `dataset`, `train_prop`, `max_epoch` and `dropout` parameters in `configs/trainer/sentence_re.yaml`. We follow the original settings in [FineCL paper](https://arxiv.org/pdf/2205.12491.pdf).

#### Hyper-parameter Seaching
- To search the best adversarial hyper-parameters for each dataset-proporation combo, you need to install [Wandb](https://github.com/wandb/wandb).
- Initialize a sweep by running:
```
wandb sweep --project <propject-name> configs/wandb.yaml
```
- Then you will get a sweep-ID, start the sweep agent by running:
```
wandb agent <sweep-ID>
```

- Or you can directly use the hyper-parameters below:
For ERICA:
|              |     | SemEval |      |     | ReTACRED |      |     | Wiki80 |      |
|--------------|-----|---------|------|-----|----------|------|-----|--------|------|
|              | 1%  | 10%     | 100% | 1%  | 10%      | 100% | 1%  | 10%    | 100% |
| adv_lr       | 0.1 | 0.05    |  0.1 | 0.1 |   0.05   |  0.1 | 0.1 |   0.1  | 0.02 |
| adv_max_norm | 0.6 |   0.2   |  0.6 | 0.6 |    0.6   |  0.4 | 0.6 |   0.4  | 0.4  |
| adv_steps    |  3  |    3    |   3  |  3  |     3    |   3  |  2  |    3   | 3    |
For FineCL (TBA)
- Revise the adversarial hyper-parameters in `configs/trainer/sentence_re.yaml` to the corresponding values and run:
```
CUDA_VISIBLE_DEVICES=0 python run.py experiment=stage_3_sentence_re.yaml
```

---

**Credits:** This work began as a fork of the [FineCL](https://github.com/wphogan/finecl) repository.
If you found our code useful, please consider citing:
```

```
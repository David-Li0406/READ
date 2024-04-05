This is the fine-tuning implementation for DocRED. 

We modify the [official code](https://github.com/thunlp/DocRED) to implement BERT-based models.

Download the [DocRED dataset](https://github.com/thunlp/DocRED/tree/master/data) and put them into the folder 'docred_data'.

Configuration files are in  `configs/`.

## Run Order:

0. Preprocess all data: `CUDA_VISIBLE_DEVICES=3,4 python run.py mode=preprocess.yaml`
1. Train stage 1 model on all data: `CUDA_VISIBLE_DEVICES=5 python run.py experiment=stage_1.yaml`
2. Run create_high_qual_training_data script to collect first-learned UIDs per first N epochs and save new 'high-quality' training data file: `python run.py mode=create_high_qual_training_set.yaml`
3. Train stage 1 model on new high-quality data

### Other scripts:

- Generate stats on first_learned UIDs: `mode=post_process.py`
- Eval a saved checkpoint: `experiment=stage_1_eval.yaml`

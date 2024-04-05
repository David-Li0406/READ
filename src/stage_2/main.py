import os
import sys
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast as autocast
from torch.utils import data
from src.transformers import AdamW, get_linear_schedule_with_warmup
from src.stage_2.dataset import Scheduler, save_data, CP_R_Dataset
from src.stage_2.model import mask_tokens, NTXentLoss_doc, NTXentLoss_wiki, CP_R


def set_seed(config):
    random.seed(config.trainer.seed)
    np.random.seed(config.trainer.seed)
    torch.manual_seed(config.trainer.seed)
    torch.cuda.manual_seed_all(config.trainer.seed)


def train(config, model, train_dataset, checkpoint_path):
    # total step
    step_tot = (len(train_dataset) // config.trainer.gradient_accumulation_steps // config.trainer.batch_size_per_gpu // config.trainer.n_gpu) * config.trainer.max_epoch
    train_sampler = data.distributed.DistributedSampler(
        train_dataset) if config.trainer.local_rank != -1 else data.RandomSampler(train_dataset)
    params = {"batch_size": config.trainer.batch_size_per_gpu, "sampler": train_sampler,
              "collate_fn": train_dataset.get_train_batch}
    train_dataloader = data.DataLoader(train_dataset, **params)

    # optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': config.trainer.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.trainer.lr, eps=config.trainer.adam_epsilon,
                      correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.trainer.warmup_steps,
                                                num_training_steps=step_tot)

    ckpt_path = os.path.join(checkpoint_path, config.trainer.save_dir, f'ckpt_of_step_{config.trainer.load_step}')
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(os.path.join(ckpt_path))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f'Load checkpoint from {ckpt_path}')

    # amp training
    if config.trainer.fp16:
        scaler = torch.cuda.amp.GradScaler()

    # distributed training
    if config.trainer.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[config.trainer.local_rank], output_device=config.trainer.local_rank,
            find_unused_parameters=True
        )

    print("Begin train...")
    print("We will train model in %d steps" % step_tot)
    global_step = config.trainer.load_step
    loss_record = []
    step_record = []
    for i in tqdm(range(config.trainer.max_epoch), ncols=100):
        tqdm_loader = tqdm(train_dataloader)
        for step, batch in enumerate(tqdm_loader):
            if len(list(batch[1].keys())) == 0:
                continue
            batch = [{k: v.cuda() for k, v in b.items()} for b in batch]
            model.train()

            if config.trainer.doc_loss == 1:
                if config.trainer.fp16:
                    with autocast():
                        m_loss_d, r_loss_d = model(batch, doc_loss=1, wiki_loss=0)
                else:
                    m_loss_d, r_loss_d = model(batch, doc_loss=1, wiki_loss=0)

                loss = m_loss_d + r_loss_d

                if config.trainer.gradient_accumulation_steps > 1:
                    loss = loss / config.trainer.gradient_accumulation_steps

                if config.trainer.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                m_loss_d = 0
                r_loss_d = 0

            if config.trainer.wiki_loss == 1:
                if config.trainer.fp16:
                    with autocast():
                        m_loss_w, r_loss_w = model(batch, doc_loss=0, wiki_loss=1)
                else:
                    m_loss_w, r_loss_w = model(batch, doc_loss=0, wiki_loss=1)

                loss = m_loss_w + r_loss_w
                if config.trainer.gradient_accumulation_steps > 1:
                    loss = loss / config.trainer.gradient_accumulation_steps
                if config.trainer.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                m_loss_w = 0
                r_loss_w = 0

            if step % config.trainer.gradient_accumulation_steps == 0:

                if config.trainer.fp16:

                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.trainer.max_grad_norm)
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = scaler.get_scale()
                    optimizer_was_run = scale_before <= scale_after
                    optimizer.zero_grad()

                    if optimizer_was_run:
                        scheduler.step()

                    global_step += 1

                else:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.trainer.max_grad_norm)
                    scheduler.step()  # Update learning rate schedule
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if config.trainer.local_rank in [0, -1] and global_step % config.trainer.log_step == 0:
                    step_record.append(global_step)
                    loss_record.append(loss)

                if config.trainer.local_rank in [0, -1] and global_step % config.trainer.save_step == 0:
                    if not os.path.exists(checkpoint_path):
                        os.mkdir(checkpoint_path)
                    if not os.path.exists(checkpoint_path + config.trainer.save_dir):
                        os.mkdir(checkpoint_path + config.trainer.save_dir)

                    ckpt = {
                        'bert-base': model.model.roberta.state_dict(),
                        'model': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }

                    if global_step > 100:
                        torch.save(ckpt, os.path.join(checkpoint_path + config.trainer.save_dir,
                                                      "ckpt_of_step_" + str(global_step)))

            if config.trainer.local_rank in [0, -1]:
                description = "step: %d, shcedule: %.3f, mlm_r: %.6f, mlm_e: %.6f, cl_r: %.6f, cl_e: %.6f" % (
                    global_step, global_step / step_tot, m_loss_d, m_loss_w, r_loss_d, r_loss_w)
                print(description)

        if config.trainer.train_sample:
            print("sampling...")
            train_dataloader.dataset.__sample__()
            print("sampled")


def train_stage_2(config):
    checkpoint_path = os.path.join(os.getcwd(), 'checkpoint')
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    if config.trainer.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(config.trainer.local_rank)
        device = torch.device("cuda", config.trainer.local_rank)
        torch.distributed.init_process_group(backend="nccl")
    set_seed(config)

    # Model and datase
    print('Preparing data')
    train_dataset = CP_R_Dataset(config)
    model = CP_R(config).to(device)

    # Barrier to make sure all process train the model simultaneously.
    if config.trainer.local_rank != -1:
        torch.distributed.barrier()
    train(config, model, train_dataset, checkpoint_path)

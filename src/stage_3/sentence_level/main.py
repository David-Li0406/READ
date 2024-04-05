import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import sys
import argparse
import sklearn.metrics
import matplotlib
import pdb
import numpy as np
import time
import random
import time
import matplotlib.pyplot as plt

matplotlib.use('Agg')
from tqdm import trange
from sklearn import metrics
from torch.utils import data
from collections import Counter
from src.transformers import AdamW, get_linear_schedule_with_warmup
from apex import amp

# Local imports
from src.stage_3.sentence_level.dataset import REDataset
from src.stage_3.sentence_level.model import REModel


def logging(s, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(os.path.join(os.path.join(os.getcwd(), 'run.log')), 'a+') as f_log:
            f_log.write(s + '\n')


def f1_score(output, label, rel_num):
    correct_by_relation = Counter()
    guess_by_relation = Counter()
    gold_by_relation = Counter()

    for i in range(len(output)):
        guess = output[i]
        gold = label[i]
        if gold == 0 and guess == 0:
            continue
        if gold == 0 and guess != 0:
            guess_by_relation[guess] += 1
        if gold != 0 and guess == 0:
            gold_by_relation[gold] += 1
        if gold != 0 and guess != 0:
            guess_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[gold] += 1

    f1_by_relation = Counter()
    recall_by_relation = Counter()
    prec_by_relation = Counter()
    for i in range(1, rel_num):
        recall = 0
        if gold_by_relation[i] > 0:
            recall = correct_by_relation[i] / gold_by_relation[i]
        precision = 0
        if guess_by_relation[i] > 0:
            precision = correct_by_relation[i] / guess_by_relation[i]
        if recall + precision > 0:
            f1_by_relation[i] = 2 * recall * precision / (recall + precision)
        recall_by_relation[i] = recall
        prec_by_relation[i] = precision

    micro_f1 = 0
    if sum(guess_by_relation.values()) != 0 and sum(correct_by_relation.values()) != 0:
        recall = sum(correct_by_relation.values()) / sum(gold_by_relation.values())
        prec = sum(correct_by_relation.values()) / sum(guess_by_relation.values())
        micro_f1 = 2 * recall * prec / (recall + prec)
    return micro_f1, f1_by_relation


def set_seed(config):
    random.seed(config.trainer.seed)
    np.random.seed(config.trainer.seed)
    torch.manual_seed(config.trainer.seed)
    torch.cuda.manual_seed_all(config.trainer.seed)


def train(config, model, train_dataloader, dev_dataloader, test_dataloader, dual_run=False):
    # total step
    step_tot = len(train_dataloader) * config.trainer.max_epoch

    # optimizer
    if config.trainer.optim == "adamw":
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
    elif config.trainer.optim == "sgd":
        params = model.parameters()
        optimizer = optim.SGD(params, config.trainer.lr)
    elif config.trainer.optim == "adam":
        params = model.parameters()
        optimizer = optim.Adam(params, config.trainer.lr)

    # amp training
    if config.trainer.optim == "adamw":
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # Data parallel
    model = nn.DataParallel(model)
    model.train()
    model.zero_grad()


    # revised
    # initialize embedding delta
    # print(model.module.bert.embeddings.word_embeddings.__dict__)
    if hasattr(model, "module"):
        vocab_size = model.module.bert.embeddings.word_embeddings.num_embeddings
    else:
        vocab_size = model.bert.embeddings.word_embeddings.num_embeddings
    delta_global_embedding = torch.zeros([vocab_size, config.trainer.hidden_size]).uniform_(-1,1)

    # 30522 bert
    # 50265 roberta
    # 21128 bert-chinese

    dims = torch.tensor([config.trainer.hidden_size]).float() # (768^(1/2))
    mag = config.trainer.adv_init_mag / torch.sqrt(dims) # 1 const (small const to init delta)
    delta_global_embedding = (delta_global_embedding * mag.view(1, 1))
    delta_global_embedding = delta_global_embedding.cuda()

    if config.trainer.dual_vocab:
        delta_global_embedding_entity = torch.zeros([vocab_size, config.trainer.hidden_size]).uniform_(-1,1)
        delta_global_embedding_entity = (delta_global_embedding_entity * mag.view(1, 1))
        delta_global_embedding_entity = delta_global_embedding_entity.cuda()


    logging("Begin train...")
    logging("We will train model in %d steps" % step_tot)
    global_step = 0
    best_dev_score = 0
    best_test_score = 0
    not_improve = 0
    for i in range(config.trainer.max_epoch):
        for batch in train_dataloader:
            inputs = {
                "input_ids": batch[0],
                "mask": batch[1],
                "h_pos": batch[2],
                "t_pos": batch[3],
                "h_pos_l": batch[6],
                "t_pos_l": batch[7],
                "label": batch[4],
                # "entity_clean_prob": batch[8],
                # "context_clean_prob": batch[9]
            }
            model.training = True
            model.train()
            
            # initialize delta
            input_ids = batch[0].cuda()
            input_ids_flat = input_ids.contiguous().view(-1)

            if isinstance(model, torch.nn.DataParallel):
                embeds_init = model.module.bert.embeddings.word_embeddings(batch[0].cuda())
            else:
                embeds_init = model.bert.embeddings.word_embeddings(batch[0].cuda())

            clean_mask = (torch.rand(embeds_init.size(0), embeds_init.size(1), device=torch.device("cuda")) < (1-config.trainer.clean_token_leaving_prob)).unsqueeze(-1).repeat(1,1,embeds_init.size(2))
            for sample_idx, (h, t, h_l, t_l) in enumerate(zip(batch[2], batch[3], batch[6], batch[7])):
                clean_mask[sample_idx, h-1:h_l+1, :] = 1
                clean_mask[sample_idx, t-1:t_l+1, :] = 1


            # embeds_init = embeds_init.clone().detach()
            input_mask = inputs['mask'].float().cuda()
            input_lengths = torch.sum(input_mask, 1) # B 

            bs,seq_len = embeds_init.size(0), embeds_init.size(1)

            #
            # delta_lb, delta_tok, total_delta = None, None, None

            dims = input_lengths * embeds_init.size(-1) # B x(768^(1/2))
            mag = config.trainer.adv_init_mag / torch.sqrt(dims) # B
            delta_lb = torch.zeros_like(embeds_init).uniform_(-1,1) * input_mask.unsqueeze(2)
            delta_lb = (delta_lb * mag.view(-1, 1, 1)).detach()


            gathered = torch.index_select(delta_global_embedding, 0, input_ids_flat) # B*seq-len D
            delta_tok = gathered.view(bs, seq_len, -1).detach() # B seq-len D

            if config.trainer.dual_vocab:
                gathered_entity = torch.index_select(delta_global_embedding_entity, 0, input_ids_flat) # B*seq-len D
                delta_tok_entity = gathered_entity.view(bs, seq_len, -1).detach() # B seq-len D
            
            denorm = torch.norm(delta_tok.view(-1,delta_tok.size(-1))).view(-1, 1, 1)
            delta_tok = delta_tok / denorm # B seq-len D  normalize delta obtained from global embedding

            # B seq-len 1

            # Adversarial-Training Loop
            for astep in range(config.trainer.adv_steps):

                # random_mask = torch.rand(embeds_init.size(), device=torch.device("cuda")) < 0.85
                             
                # craft input embedding
                delta_lb.requires_grad_()
                delta_tok.requires_grad_()

                inputs_embeds = embeds_init + (delta_lb + delta_tok) * clean_mask
                inputs['inputs_embeds'] = inputs_embeds
                loss, output = model(**inputs)

                loss = loss / config.trainer.adv_steps
                
                if torch.isnan(loss):
                    return best_test_score, best_dev_score

                # tr_loss += loss.item()
                if config.trainer.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward(retain_graph=True)
                else:
                    loss.backward(retain_graph=True)

                if astep == config.trainer.adv_steps - 1:
                    # further updates on delta

                    delta_tok = delta_tok.detach()
                    if not config.trainer.dual_vocab:
                        delta_global_embedding = delta_global_embedding.index_put_((input_ids_flat,), delta_tok, True)
                    else:
                        for sample_idx, (h, t, h_l, t_l) in enumerate(zip(batch[2], batch[3], batch[6], batch[7])):
                            input_ids_sample = input_ids[sample_idx]
                            input_ids_sample_entity_idx = [idx for idx in range(h, h_l)] + [idx for idx in range(t, t_l)]
                            input_ids_sample_context_idx = [idx for idx in range(0, input_ids_sample.size(0)) if not ((idx>=h and idx<h_l)  or (idx>=t and idx<t_l))]
                            input_ids_sample_entity = input_ids_sample[input_ids_sample_entity_idx]
                            input_ids_sample_context = input_ids_sample[input_ids_sample_context_idx]
                            delta_tok_context = delta_tok[sample_idx, input_ids_sample_context_idx, :]
                            delta_tok_entity = delta_tok[sample_idx, input_ids_sample_entity_idx, :]
                            delta_global_embedding = delta_global_embedding.index_put_((input_ids_sample_context.contiguous().view(-1),), delta_tok_context, True)
                            delta_global_embedding_entity = delta_global_embedding_entity.index_put_((input_ids_sample_entity.contiguous().view(-1),), delta_tok_entity, True)

                    break

                # 2) get grad on delta
                if delta_lb is not None:
                    delta_lb_grad = delta_lb.grad.clone().detach()
                if delta_tok is not None:
                    delta_tok_grad = delta_tok.grad.clone().detach()


                # 3) update and clip

                    
                denorm_lb = torch.norm(delta_lb_grad.view(bs, -1), dim=1).view(-1, 1, 1)
                denorm_lb = torch.clamp(denorm_lb, min=1e-8)
                denorm_lb = denorm_lb.view(bs, 1, 1)


                denorm_tok = torch.norm(delta_tok_grad, dim=-1) # B seq-len 
                denorm_tok = torch.clamp(denorm_tok, min=1e-8)
                denorm_tok = denorm_tok.view(bs, seq_len, 1) # B seq-len 1


                delta_lb = (delta_lb + config.trainer.adv_lr * delta_lb_grad / denorm_lb).detach()
                delta_tok = (delta_tok + config.trainer.adv_lr * delta_tok_grad / denorm_tok).detach()

                # calculate clip

                delta_norm_tok = torch.norm(delta_tok, p=2, dim=-1).detach() # B seq-len
                mean_norm_tok, _ = torch.max(delta_norm_tok, dim=-1, keepdim=True) # B,1 
                reweights_tok = (delta_norm_tok / mean_norm_tok).view(bs, seq_len, 1) # B seq-len, 1

                delta_tok = delta_tok * reweights_tok

                total_delta = delta_tok + delta_lb

                delta_norm = torch.norm(total_delta.view(bs, -1).float(), p=2, dim=1).detach()
                exceed_mask = (delta_norm > config.trainer.adv_max_norm).to(embeds_init)
                reweights = (config.trainer.adv_max_norm / delta_norm * exceed_mask \
                                + (1-exceed_mask)).view(-1, 1, 1) # B 1 1

                # clip

                delta_lb = (delta_lb * reweights).detach()
                delta_tok = (delta_tok * reweights).detach()


            # *************************** END *******************

            if config.trainer.optim == "adamw":
                nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.trainer.max_grad_norm)
            optimizer.step()
            if config.trainer.optim == "adamw":
                scheduler.step()
            model.zero_grad()
            global_step += 1

            output = output.cpu().detach().numpy()
            label = batch[4].numpy()
            crr = (output == label).sum()
            tot = label.shape[0]

            sys.stdout.write("epoch: %d, loss: %.6f, acc: %.3f\r" % (i, loss, crr / tot))
            sys.stdout.flush()

        # dev
        with torch.no_grad():
            logging("")
            logging("deving....")
            model.training = False
            model.eval()

            if "semeval" in config.trainer.dataset or "tacred" in config.trainer.dataset:
                eval_func = eval_F1
            elif config.trainer.dataset == "wiki80" or config.trainer.dataset == "chemprot":
                eval_func = eval_ACC

            score = eval_func(config, model, dev_dataloader)
            if score > best_dev_score:
                not_improve = 0
                best_dev_score = score
                best_test_score = eval_func(config, model, test_dataloader)
                logging("Best Dev score: %.3f,\tTest score: %.3f" % (best_dev_score, best_test_score))
            else:
                not_improve+=1
                logging("Dev score: %.3f" % score)
            logging("-----------------------------------------------------------")

            # if not_improve == 20:
            #     return best_test_score, best_dev_score

    logging("@RESULT: " + config.trainer.dataset + " Test score is %.3f" % best_test_score)

    # File name settings:
    model_name = config.model_name_or_path
    fname_out = f're_{config.trainer.dataset}_{model_name}_{str(config.trainer.train_prop)}_{config.trainer.seed}_{best_test_score:.3f}.log'

    f = open(os.path.join(os.getcwd(), fname_out), 'a+')
    if config.pretrained_model_path == "None":
        f.write("bert-base\t" + config.trainer.dataset + "\t" + str(
            config.trainer.train_prop) + "\t" + config.trainer.mode + "\t" + "seed:" + str(
            config.trainer.seed) + "\t" + "max_epoch:" + str(config.trainer.max_epoch) + "\t" + str(
            time.ctime()) + "\n")
    else:
        f.write(config.pretrained_model_path + "\t" + config.trainer.dataset + "\t" + str(
            config.trainer.train_prop) + "\t" + config.trainer.mode + "\t" + "seed:" + str(
            config.trainer.seed) + "\t" + "max_epoch:" + str(config.trainer.max_epoch) + "\t" + str(
            time.ctime()) + "\n")
    f.write("@RESULT: Best Dev score is %.3f, Test score is %.3f\n" % (best_dev_score, best_test_score))
    f.write("--------------------------------------------------------------\n")
    f.close()
    # torch.save(model.state_dict(), '/data/dawei/finecl_WAT/{}_best_wat_{}_no_spv.bin'.format(config.trainer.dataset, config.trainer.clean_token_leaving_prob))
    # if dual_run:
    return best_test_score, best_dev_score


def eval_F1(config, model, dataloader):
    tot_label = []
    tot_output = []
    for batch in dataloader:
        inputs = {
            "input_ids": batch[0],
            "mask": batch[1],
            "h_pos": batch[2],
            "t_pos": batch[3],
            "h_pos_l": batch[6],
            "t_pos_l": batch[7],
            "label": batch[4]
        }
        _, output = model(**inputs)
        tot_label.extend(batch[4].tolist())
        tot_output.extend(output.cpu().detach().tolist())

    f1, _ = f1_score(tot_output, tot_label, config.trainer.rel_num)
    return f1


def eval_ACC(config, model, dataloader):
    tot = 0.0
    crr = 0.0
    for batch in dataloader:
        inputs = {
            "input_ids": batch[0],
            "mask": batch[1],
            "h_pos": batch[2],
            "t_pos": batch[3],
            "h_pos_l": batch[6],
            "t_pos_l": batch[7],
            "label": batch[4]
        }
        _, output = model(**inputs)
        output = output.cpu().detach().numpy()
        label = batch[4].numpy()
        crr += (output == label).sum()
        tot += label.shape[0]

        sys.stdout.write("acc: %.3f\r" % (crr / tot))
        sys.stdout.flush()

    return crr / tot


def train_stage_3_sentence_re(config):
    logging('Experiment directory: ', os.getcwd())
    set_seed(config)

    dataset_dir = config.trainer.data_dir + config.trainer.dataset
    logging(f"Using {config.trainer.train_prop * 100}% train data from {dataset_dir}!")
    train_set = REDataset(dataset_dir, f"train_{config.trainer.train_prop}.json", config)
    dev_set = REDataset(dataset_dir, "dev.json", config)
    test_set = REDataset(dataset_dir, "test.json", config)

    logging('loading dataloader...')
    train_dataloader = data.DataLoader(train_set, batch_size=config.trainer.batch_size_per_gpu, shuffle=True)
    dev_dataloader = data.DataLoader(dev_set, batch_size=config.trainer.batch_size_per_gpu*2, shuffle=False)
    test_dataloader = data.DataLoader(test_set, batch_size=config.trainer.batch_size_per_gpu*2, shuffle=False)

    rel2id = json.load(open(os.path.join(dataset_dir, "rel2id.json")))
    config.trainer.rel_num = len(rel2id)

    logging('loading model...')
    model = REModel(config)
    model.cuda()
    best_test_score, best_dev_score = train(config, model, train_dataloader, dev_dataloader, test_dataloader)
    logging('Results directory: ', os.getcwd())

    del model
    torch.cuda.empty_cache()

    return best_test_score, best_dev_score

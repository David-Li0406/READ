import os
import csv
import json
import time
import torch
import pickle
import random
import argparse
import matplotlib
import numpy as np
import sklearn.metrics
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import date, datetime
from collections import defaultdict
from apex import amp
from torch.optim.lr_scheduler import LambdaLR
from src.transformers import (AdamW, BertModel, RobertaModel)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from src.stage_1.re_model import REModel

matplotlib.use('Agg')

MODEL_CLASSES = {
    'bert': BertModel,
    'roberta': RobertaModel,
}

IGNORE_INDEX = -100


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """

    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class REDataset():
    def __init__(self, prefix, data_path, h_t_limit, config):
        self.h_t_limit = h_t_limit
        if config.use_erica_data:
            assert config.training_data_type == 'distant'
            self.data_path = config.prepro_erica_training_data_dir
        else:
            self.data_path = data_path

        if config.trainer.ratio < 1 and prefix == 'train':
            self.train_file = json.load(
                open(os.path.join(self.data_path, prefix + '_' + str(config.trainer.ratio) + '.json')))
            self.data_train_bert_token = np.load(os.path.join(self.data_path, prefix + '_' + str(
                config.trainer.ratio) + f'_{config.model_type}_token.npy'))
            self.data_train_bert_mask = np.load(os.path.join(self.data_path, prefix + '_' + str(
                config.trainer.ratio) + f'_{config.model_type}_mask.npy'))
            self.data_train_bert_starts_ends = np.load(os.path.join(self.data_path, prefix + '_' + str(
                config.trainer.ratio) + f'_{config.model_type}_starts_ends.npy'))
        else:
            train_fname = ''  # placeholder
            if prefix == 'train':
                if 'docred_' not in self.data_path:  # data from ERICA repo has different name for annotated data
                    prefix = prefix + '_' + config.training_data_type  # use distant OR annotated training data

                if config.reduced_data:
                    prefix = prefix + '_' + str(config.train_prop)

                # Logic to load numbered ERICA train_distant data files
                if config.use_erica_data and config.training_data_type == 'distant':
                    prefix = prefix + '_' + str(config.erica_file_num)

                train_fname = os.path.join(self.data_path, prefix + '.json')


            print('Training file: ', train_fname)
            self.train_file = json.load(open(train_fname))

            print(f'Preprocessed files: {self.data_path}, {prefix}, {config.model_type}')
            self.data_train_bert_token = np.load(
                os.path.join(self.data_path, prefix + f'_{config.model_type}_token.npy'))
            self.data_train_bert_mask = np.load(
                os.path.join(self.data_path, prefix + f'_{config.model_type}_mask.npy'))
            self.data_train_bert_starts_ends = np.load(
                os.path.join(self.data_path, prefix + f'_{config.model_type}_starts_ends.npy'))

    def __getitem__(self, index):
        return self.train_file[index], self.data_train_bert_token[index], \
               self.data_train_bert_mask[index], self.data_train_bert_starts_ends[index]

    def __len__(self):
        return self.data_train_bert_token.shape[0]


class Accuracy(object):
    def __init__(self):
        self.correct = 0
        self.total = 0

    def add(self, is_correct):
        self.total += 1
        if is_correct:
            self.correct += 1

    def get(self):
        if self.total == 0:
            return 0.0
        else:
            return float(self.correct) / self.total

    def clear(self):
        self.correct = 0
        self.total = 0


class Controller(object):
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.acc_NA = Accuracy()
        self.acc_not_NA = Accuracy()
        self.acc_total = Accuracy()
        self.config = config
        self.max_seq_length = config.trainer.max_seq_length
        self.relation_num = 97
        self.max_epoch = config.num_train_epochs
        self.evaluate_during_training_epoch = config.trainer.evaluate_during_training_epoch
        self.log_period = config.trainer.logging_steps
        self.neg_multiple = 3  # The number of negative examples sampled is three times that of positive examples
        self.warmup_ratio = 0.1
        self.training_data_type = config.training_data_type

        self.data_path = config.prepro_data_dir
        self.use_docred_from_erica_authors = ('docred_' in self.data_path)
        # if debug mode, load debug data and reduce epochs
        if config.debug_mode:
            self.data_path = os.path.join(config.data_dir, 'DocRED_debug_preprocessed')
            self.max_epoch = config.num_train_epochs_debug
        print('Loading dev/test data from ', self.data_path)

        if config.use_erica_data:
            print('Loading training data from ', config.prepro_erica_training_data_dir)
            self.relation_num = 1040  # number of rel classes in ERICA pretrain data
        else:
            print('Loading training data from ', self.data_path)

        self.dir_first_learned = os.path.join(os.getcwd(), "first_learned")
        self.dir_first_learned_train = os.path.join(self.dir_first_learned, 'train_distant_fl')
        self.dir_dev_preds = os.path.join('dev_final_preds')

        self.batch_size = config.trainer.batch_size
        self.gradient_accumulation_steps = config.trainer.gradient_accumulation_steps
        self.lr = config.trainer.learning_rate

        self.h_t_limit = 1800  # The maximum number of relation facts

        self.test_batch_size = self.batch_size * 2
        self.test_relation_limit = self.h_t_limit

        self.dis2idx = np.zeros((512), dtype='int64')
        self.dis2idx[1] = 1
        self.dis2idx[2:] = 2
        self.dis2idx[4:] = 3
        self.dis2idx[8:] = 4
        self.dis2idx[16:] = 5
        self.dis2idx[32:] = 6
        self.dis2idx[64:] = 7
        self.dis2idx[128:] = 8
        self.dis2idx[256:] = 9
        self.dis_size = 20

        self.train_prefix = config.trainer.train_prefix
        self.test_prefix = config.trainer.test_prefix

        self.checkpoint_dir = os.path.join(os.getcwd(), 'ce_checkpoint')
        self.fig_result_dir = os.path.join(os.getcwd(), 'ce_fig_result')
        self.log_dir = os.path.join(os.getcwd(), 'ce_log')
        self.best_model_path = os.path.join(self.checkpoint_dir,
                                            f"{self.config.model_type}_best.bin")

        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        if 'eval' not in config.name:  # these dirs are only needed during training
            if not os.path.exists(self.dir_first_learned):
                os.mkdir(self.dir_first_learned)
                os.mkdir(self.dir_first_learned_train)

            if not os.path.exists(self.checkpoint_dir):
                os.mkdir(self.checkpoint_dir)

            if not os.path.exists(self.fig_result_dir):
                os.mkdir(self.fig_result_dir)
        else:
            if self.config.error_analysis:
                if not os.path.exists(self.dir_dev_preds): os.mkdir(self.dir_dev_preds)

        # QA for the runs
        if config.ignore_dev_set:
            assert config.training_data_type == 'distant'
            assert not config.load_pretrained_checkpoint

    def load_data(self):
        self.rel2id = json.load(open(os.path.join(self.data_path, 'rel2id.json')))
        self.id2rel = {v: k for k, v in self.rel2id.items()}

        if self.config.use_erica_data and self.config.training_data_type == 'distant':
            prefix = 'train_distant_' + str(self.config.erica_file_num)
            self.data_file = json.load(open(os.path.join(self.data_path, prefix + '.json')))
        else:
            prefix = self.test_prefix
            self.is_test = ('test' == prefix)
            self.data_file = json.load(open(os.path.join(self.data_path, prefix + '.json')))

        self.data_bert_token = np.load(
            os.path.join(self.data_path, prefix + f'_{self.config.model_type}_token.npy'))
        self.data_bert_mask = np.load(
            os.path.join(self.data_path, prefix + f'_{self.config.model_type}_mask.npy'))
        self.data_bert_starts_ends = np.load(
            os.path.join(self.data_path, prefix + f'_{self.config.model_type}_starts_ends.npy'))

        self.data_len = self.data_bert_token.shape[0]
        assert (self.data_len == len(self.data_file))

        self.test_batches = self.data_bert_token.shape[0] // self.test_batch_size
        if self.data_bert_token.shape[0] % self.test_batch_size != 0:
            self.test_batches += 1

        self.test_order = list(range(self.data_len))
        self.test_order.sort(key=lambda x: np.sum(self.data_bert_token[x] > 0), reverse=True)

    def get_test_batch(self):
        context_idxs = torch.LongTensor(self.test_batch_size, self.max_seq_length).to(self.device)
        h_mapping = torch.Tensor(self.test_batch_size, self.test_relation_limit, self.max_seq_length).to(self.device)
        t_mapping = torch.Tensor(self.test_batch_size, self.test_relation_limit, self.max_seq_length).to(self.device)
        relation_label = torch.LongTensor(self.test_batch_size, self.h_t_limit).fill_(IGNORE_INDEX)
        relation_uid = torch.LongTensor(self.test_batch_size, self.h_t_limit).fill_(IGNORE_INDEX)

        relation_mask = torch.Tensor(self.test_batch_size, self.h_t_limit).to(self.device)
        ht_pair_pos = torch.LongTensor(self.test_batch_size, self.h_t_limit).to(self.device)

        context_masks = torch.LongTensor(self.test_batch_size, self.max_seq_length).to(self.device)

        for b in range(self.test_batches):
            start_id = b * self.test_batch_size
            cur_bsz = min(self.test_batch_size, self.data_len - start_id)
            cur_batch = list(self.test_order[start_id: start_id + cur_bsz])

            for mapping in [h_mapping, t_mapping, relation_mask]:
                mapping.zero_()

            ht_pair_pos.zero_()

            max_h_t_cnt = 1

            labels = []
            labels_multi = []

            L_vertex = []
            titles = []
            indexes = []

            all_test_idxs = []
            all_test_idxs_0 = []
            all_test_idxs_1 = []
            all_test_idxs_2 = []
            all_test_idxs_multi = []
            for i, index in enumerate(cur_batch):
                context_idxs[i].copy_(torch.from_numpy(self.data_bert_token[index, :]))
                context_masks[i].copy_(torch.from_numpy(self.data_bert_mask[index, :]))

                idx2label = defaultdict(list)
                ins = self.data_file[index]
                starts_pos = self.data_bert_starts_ends[index, :, 0]
                ends_pos = self.data_bert_starts_ends[index, :, 1]
                trip2uid = {}
                for label in ins['labels']:
                    idx2label[(label['h'], label['t'])].append(label['r'])
                    if self.use_docred_from_erica_authors:
                        trip2uid[(label['h'], label['t'], label['r'])] = 0
                    else:
                        trip2uid[(label['h'], label['t'], label['r'])] = label['uid']

                L = len(ins['vertexSet'])
                titles.append(ins['title'])

                j = 0
                test_idxs = []

                test_idxs_0 = []
                test_idxs_1 = []
                test_idxs_2 = []
                test_idxs_multi = []
                for h_idx in range(L):
                    for t_idx in range(L):
                        if h_idx != t_idx:
                            hlist = ins['vertexSet'][h_idx]
                            tlist = ins['vertexSet'][t_idx]

                            hlist = [(starts_pos[h['pos'][0]], ends_pos[h['pos'][1] - 1]) for h in hlist if
                                     ends_pos[h['pos'][1] - 1] < 511]
                            tlist = [(starts_pos[t['pos'][0]], ends_pos[t['pos'][1] - 1]) for t in tlist if
                                     ends_pos[t['pos'][1] - 1] < 511]
                            if len(hlist) == 0 or len(tlist) == 0:
                                continue

                            for h in hlist:
                                h_mapping[i, j, h[0]:h[1]] = 1.0 / len(hlist) / (h[1] - h[0])

                            for t in tlist:
                                t_mapping[i, j, t[0]:t[1]] = 1.0 / len(tlist) / (t[1] - t[0])

                            relation_mask[i, j] = 1
                            label = idx2label[(h_idx, t_idx)]
                            if len(label):
                                rt = np.random.randint(len(label))
                                relation_label[i, j] = label[rt]
                                relation_uid[i, j] = trip2uid[(h_idx, t_idx, label[rt])]
                            else:
                                relation_label[i, j] = 0
                                relation_uid[i, j] = 0

                            delta_dis = hlist[0][0] - tlist[0][0]
                            if delta_dis < 0:
                                ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                            else:
                                ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])

                            test_idxs.append((h_idx, t_idx))
                            j += 1

                max_h_t_cnt = max(max_h_t_cnt, j)
                label_set = {}
                label_uids = {}
                for label in ins['labels']:
                    label_set[(label['h'], label['t'], label['r'])] = label['in_annotated_train']
                    if self.use_docred_from_erica_authors:
                        label_uids[(label['h'], label['t'], label['r'])] = 0
                    else:
                        label_uids[(label['h'], label['t'], label['r'])] = label['uid']

                for label in ins['labels']:
                    label_set[(label['h'], label['t'], label['r'])] = label['in_annotated_train']

                label_multi_set = {}
                for label in ins['labels']:
                    if len(label['evidence']) > 1:
                        test_idxs_2.append((label['h'], label['t']))
                        label_multi_set[(label['h'], label['t'])] = 2
                    elif len(label['evidence']) == 1:
                        test_idxs_1.append((label['h'], label['t']))
                        label_multi_set[(label['h'], label['t'])] = 1
                    elif len(label['evidence']) == 0:
                        test_idxs_0.append((label['h'], label['t']))
                        label_multi_set[(label['h'], label['t'])] = 0

                    hlist = [x['sent_id'] for x in ins['vertexSet'][label['h']]]
                    tlist = [x['sent_id'] for x in ins['vertexSet'][label['t']]]
                    flag = 0
                    for evi_idx in label['evidence']:
                        if evi_idx in hlist and evi_idx in tlist:
                            flag = 1
                    if flag == 0 and len(label['evidence']) > 1:
                        test_idxs_multi.append((label['h'], label['t']))

                labels.append(label_set)
                labels_multi.append(label_multi_set)

                L_vertex.append(L)
                indexes.append(index)
                all_test_idxs.append(test_idxs)

                all_test_idxs_0.append(test_idxs_0)
                all_test_idxs_1.append(test_idxs_1)
                all_test_idxs_2.append(test_idxs_2)
                all_test_idxs_multi.append(test_idxs_multi)

            max_c_len = self.max_seq_length

            yield {'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                   'h_mapping': h_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   't_mapping': t_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   'relation_label': relation_label[:, :max_h_t_cnt].contiguous(),
                   'relation_uid': relation_uid[:, :max_h_t_cnt].contiguous(),
                   'labels': labels,
                   'label_multi': labels_multi,
                   'L_vertex': L_vertex,
                   'titles': titles,
                   'ht_pair_pos': ht_pair_pos[:cur_bsz, :max_h_t_cnt],
                   'indexes': indexes,
                   'label_uids': label_uids,
                   'context_masks': context_masks[:cur_bsz, :max_c_len].contiguous(),
                   'all_test_idxs': all_test_idxs,
                   'all_test_idxs_0': all_test_idxs_0,
                   'all_test_idxs_1': all_test_idxs_1,
                   'all_test_idxs_2': all_test_idxs_2,
                   'all_test_idxs_multi': all_test_idxs_multi,
                   }

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def get_train_batch(self, batch):
        batch_size = len(batch)
        max_length = self.max_seq_length
        h_t_limit = self.h_t_limit
        relation_num = self.relation_num
        context_idxs = torch.LongTensor(batch_size, max_length).zero_()
        h_mapping = torch.Tensor(batch_size, h_t_limit, max_length).zero_()
        t_mapping = torch.Tensor(batch_size, h_t_limit, max_length).zero_()
        relation_multi_label = torch.Tensor(batch_size, h_t_limit, relation_num).zero_()
        relation_mask = torch.Tensor(batch_size, h_t_limit).zero_()

        context_masks = torch.LongTensor(batch_size, self.max_seq_length).zero_()
        ht_pair_pos = torch.LongTensor(batch_size, h_t_limit).zero_()

        relation_label = torch.LongTensor(batch_size, h_t_limit).fill_(IGNORE_INDEX)
        relation_uid = torch.LongTensor(batch_size, h_t_limit).fill_(IGNORE_INDEX)

        for i, item in enumerate(batch):
            max_h_t_cnt = 1

            context_idxs[i].copy_(torch.from_numpy(item[1]))
            context_masks[i].copy_(torch.from_numpy(item[2]))
            starts_pos = item[3][:, 0]
            ends_pos = item[3][:, 1]

            ins = item[0]

            labels = ins['labels']
            idx2label = defaultdict(list)
            trip2uid = {}

            for label in labels:
                idx2label[(label['h'], label['t'])].append(label['r'])
                if self.use_docred_from_erica_authors:
                    trip2uid[(label['h'], label['t'], label['r'])] = 0
                else:
                    trip2uid[(label['h'], label['t'], label['r'])] = label['uid']

            train_tripe = list(idx2label.keys())
            j = 0
            for (h_idx, t_idx) in train_tripe:  # for each triple in this batch
                if j == self.h_t_limit:
                    break
                hlist = ins['vertexSet'][h_idx]
                tlist = ins['vertexSet'][t_idx]

                hlist = [(starts_pos[h['pos'][0]], ends_pos[h['pos'][1] - 1]) for h in hlist if
                         ends_pos[h['pos'][1] - 1] < 511]
                tlist = [(starts_pos[t['pos'][0]], ends_pos[t['pos'][1] - 1]) for t in tlist if
                         ends_pos[t['pos'][1] - 1] < 511]

                if len(hlist) == 0 or len(tlist) == 0:
                    continue

                for h in hlist:
                    h_mapping[i, j, h[0]:h[1]] = 1.0 / len(hlist) / (h[1] - h[0])

                for t in tlist:
                    t_mapping[i, j, t[0]:t[1]] = 1.0 / len(tlist) / (t[1] - t[0])

                label = idx2label[(h_idx, t_idx)]

                delta_dis = hlist[0][0] - tlist[0][0]
                if delta_dis < 0:
                    ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                else:
                    ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])

                for r in label:
                    relation_multi_label[i, j, r] = 1

                relation_mask[i, j] = 1
                rt = np.random.randint(len(label))
                relation_label[i, j] = label[rt]
                relation_uid[i, j] = trip2uid[(h_idx, t_idx, label[rt])]

                j += 1

            lower_bound = min(len(ins['na_triple']), len(train_tripe) * self.neg_multiple)
            sel_idx = random.sample(list(range(len(ins['na_triple']))), lower_bound)
            sel_ins = [ins['na_triple'][s_i] for s_i in sel_idx]

            for (h_idx, t_idx) in sel_ins:
                if j == h_t_limit:
                    break
                hlist = ins['vertexSet'][h_idx]
                tlist = ins['vertexSet'][t_idx]

                hlist = [(starts_pos[h['pos'][0]], ends_pos[h['pos'][1] - 1]) for h in hlist if
                         ends_pos[h['pos'][1] - 1] < 511]
                tlist = [(starts_pos[t['pos'][0]], ends_pos[t['pos'][1] - 1]) for t in tlist if
                         ends_pos[t['pos'][1] - 1] < 511]
                if len(hlist) == 0 or len(tlist) == 0:
                    continue

                for h in hlist:
                    h_mapping[i, j, h[0]:h[1]] = 1.0 / len(hlist) / (h[1] - h[0])

                for t in tlist:
                    t_mapping[i, j, t[0]:t[1]] = 1.0 / len(tlist) / (t[1] - t[0])

                delta_dis = hlist[0][0] - tlist[0][0]

                relation_multi_label[i, j, 0] = 1
                relation_label[i, j] = 0
                relation_uid[i, j] = -1
                relation_mask[i, j] = 1

                if delta_dis < 0:
                    ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                else:
                    ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])
                j += 1

            max_h_t_cnt = max(max_h_t_cnt, len(train_tripe) + lower_bound)

        return {'context_idxs': context_idxs,
                'h_mapping': h_mapping[:, :max_h_t_cnt, :].contiguous(),
                't_mapping': t_mapping[:, :max_h_t_cnt, :].contiguous(),
                'relation_label': relation_label[:, :max_h_t_cnt].contiguous(),
                'relation_uid': relation_uid[:, :max_h_t_cnt].contiguous(),
                'relation_multi_label': relation_multi_label[:, :max_h_t_cnt].contiguous(),
                'relation_mask': relation_mask[:, :max_h_t_cnt].contiguous(),
                'ht_pair_pos': ht_pair_pos[:, :max_h_t_cnt].contiguous(),
                'context_masks': context_masks,
                }

    def eval_train(self, model, epoch, config, all_learned_train):
        '''
        Function used to evaluate entire training set after each epoch.
        e.g. "Epoch-based" learning order verus "Batch-based" learning order
        '''
        train_dataset = REDataset(self.train_prefix, self.data_path, self.h_t_limit, config)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.batch_size,
                                      collate_fn=self.get_train_batch, num_workers=2)

        # Start training
        model.eval()
        print(f'Evaluating epoch: {epoch}')
        learned_this_epoch = defaultdict(set)
        preds_this_epoch = defaultdict(set)
        self.acc_NA.clear()
        self.acc_not_NA.clear()
        self.acc_total.clear()

        for batch in train_dataloader:
            data = {k: v.to(self.device) for k, v in batch.items()}

            context_idxs = data['context_idxs']
            h_mapping = data['h_mapping']
            t_mapping = data['t_mapping']
            relation_label = data['relation_label']
            # relation_multi_label = data['relation_multi_label']
            relation_mask = data['relation_mask']

            relation_uid = data['relation_uid']

            ht_pair_pos = data['ht_pair_pos']
            context_masks = data['context_masks']

            if torch.sum(relation_mask) == 0:
                print('zero input')
                continue

            dis_h_2_t = ht_pair_pos + 10
            dis_t_2_h = -ht_pair_pos + 10

            predict_re = model(context_idxs, h_mapping, t_mapping, dis_h_2_t, dis_t_2_h, context_masks)
            output = torch.argmax(predict_re, dim=-1)
            output = output.data.cpu().numpy()
            relation_label = relation_label.data.cpu().numpy()

            for i in range(output.shape[0]):
                for j in range(output.shape[1]):
                    label = relation_label[i][j]
                    uid = relation_uid[i][j].item()
                    if label < 0:
                        break
                    if label == 0:
                        self.acc_NA.add(output[i][j] == label)
                    else:
                        self.acc_not_NA.add(output[i][j] == label)
                        preds_this_epoch[epoch].add((output[i][j], label, uid))
                        if output[i][j] == label and uid not in all_learned_train:
                            all_learned_train.add(uid)
                            learned_this_epoch[epoch].add(uid)

                    self.acc_total.add(output[i][j] == label)

        self.test_prefix = 'train_distant'
        self.save_first_learned(learned_this_epoch, epoch)  # save training first learned
        return all_learned_train

    def train(self, model_type, model_name_or_path, save_name, config):
        self.load_data()
        train_dataset = REDataset(self.train_prefix, self.data_path, self.h_t_limit, config)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.batch_size,
                                      collate_fn=self.get_train_batch, num_workers=2)
        bert_model = MODEL_CLASSES[model_type].from_pretrained(model_name_or_path)

        if config.load_pretrained_checkpoint:
            assert config.training_data_type == 'annotated'
            print("Path to checkpoint: ", config.pretrain_checkpoint)
            ckpt = torch.load(config.pretrain_checkpoint)
            bert_model.load_state_dict(ckpt["bert-base"])

        model = REModel(config=self, bert_model=bert_model)
        model.to(self.device)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.config.trainer.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, eps=self.config.trainer.adam_epsilon)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        if self.config.debug_mode:
            tot_step = 10
            print('debug mode is active...')
        else:
            tot_step = int(
                (len(train_dataset) // self.batch_size + 1) / self.gradient_accumulation_steps * self.max_epoch)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=int(self.warmup_ratio * tot_step), t_total=tot_step)
        save_step = int((
                                len(train_dataset) // self.batch_size + 1) / self.gradient_accumulation_steps * self.evaluate_during_training_epoch)
        print("tot_step:", tot_step, "save_step:", save_step, self.lr)

        BCE = nn.BCEWithLogitsLoss(reduction='none')

        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        best_all_f1_d = 0.0
        best_dev_theta = -1
        best_result = None
        best_all_epoch = 0

        model.train()

        global_step = 0
        total_loss = 0
        start_time = time.time()

        def logging(s, print_=True, log_=True):
            if print_:
                print(s)
            if log_:
                with open(os.path.join(os.path.join(self.log_dir, save_name + '.log')), 'a+') as f_log:
                    f_log.write(s + '\n')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim(0.3, 1.0)
        plt.xlim(0.0, 0.4)
        plt.title('Precision-Recall')
        plt.grid(True)
        step = 0

        # Set to keep track of which triples (UIDs) have been learned
        all_learned_train = set()
        all_learned_dev = set()

        # Start training
        for epoch in range(self.max_epoch):
            learned_this_epoch = defaultdict(set)
            preds_this_epoch = defaultdict(set)
            self.acc_NA.clear()
            self.acc_not_NA.clear()
            self.acc_total.clear()

            for batch in train_dataloader:
                data = {k: v.to(self.device) for k, v in batch.items()}

                context_idxs = data['context_idxs']
                h_mapping = data['h_mapping']
                t_mapping = data['t_mapping']
                relation_label = data['relation_label']
                relation_multi_label = data['relation_multi_label']
                relation_mask = data['relation_mask']

                relation_uid = data['relation_uid']

                ht_pair_pos = data['ht_pair_pos']
                context_masks = data['context_masks']

                if torch.sum(relation_mask) == 0:
                    print('zero input')
                    continue

                dis_h_2_t = ht_pair_pos + 10
                dis_t_2_h = -ht_pair_pos + 10

                predict_re = model(context_idxs, h_mapping, t_mapping, dis_h_2_t, dis_t_2_h, context_masks)

                pred_loss = BCE(predict_re, relation_multi_label) * relation_mask.unsqueeze(2)
                loss = torch.sum(pred_loss) / (self.relation_num * torch.sum(relation_mask))

                if torch.isnan(loss):
                    pickle.dump(data, open("crash_data.pkl", "wb"))
                    path = os.path.join(self.checkpoint_dir, model_type + "_crash")
                    torch.save(model.state_dict(), path)
                    logging('Loss is NAN error.')

                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                output = torch.argmax(predict_re, dim=-1)
                output = output.data.cpu().numpy()
                loss.backward()
                relation_label = relation_label.data.cpu().numpy()

                if not config.epoch_based_learning_order:
                    for i in range(output.shape[0]):
                        for j in range(output.shape[1]):
                            label = relation_label[i][j]
                            uid = relation_uid[i][j].item()
                            if label < 0:
                                break
                            if label == 0:
                                self.acc_NA.add(output[i][j] == label)
                            else:
                                self.acc_not_NA.add(output[i][j] == label)
                                preds_this_epoch[epoch].add((output[i][j], label, uid))  # tuple: (pred, label, uid)
                                if output[i][j] == label and uid not in all_learned_train:
                                    all_learned_train.add(uid)
                                    learned_this_epoch[epoch].add(uid)
                            self.acc_total.add(output[i][j] == label)

                total_loss += loss.item()
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if global_step % self.log_period == 0:
                        cur_loss = total_loss / self.log_period
                        elapsed = time.time() - start_time
                        logging(
                            '| epoch {:2d} | step {:4d} |  ms/b {:5.2f} | train loss {:.8f} | NA acc: {:4.2f} | not NA acc: {:4.2f}  | tot acc: {:4.2f} '.format(
                                epoch, global_step, elapsed * 1000 / self.log_period, cur_loss, self.acc_NA.get(),
                                self.acc_not_NA.get(), self.acc_total.get()))
                        total_loss = 0
                        start_time = time.time()
                step += 1
                # END BATCH LOOP

            # Every epoch, save learned TRAIN trips (distantly labeled data only)
            if config.epoch_based_learning_order:
                new_all_learned = self.eval_train(model, epoch, config, all_learned_train)
                all_learned_train.update(new_all_learned)
            elif config.training_data_type == 'distant':
                self.save_first_learned(learned_this_epoch, epoch)  # save training first learned

            # Assess dev performance
            if not config.use_erica_data and epoch > 0.5 * self.max_epoch:
                logging('-' * 89)
                model.eval()
                logging('Dev set evaluation:')
                self.test_prefix = 'dev'
                self.load_data()
                all_f1_d, test_f1_d, ign_f1_d, f1_d, auc_d, pr_x_d, pr_y_d, input_theta_dev, all_learned_dev, new_theta = self.eval(
                    model, save_name, epoch, save_learn_order=False,
                    all_learned_uids=all_learned_dev)  # save dev first learned

                # Save best pretraining model:
                if config.ignore_dev_set and epoch == self.max_epoch - 1:  # last epoch
                    assert config.training_data_type == 'distant'
                    logging('Recording END model.')
                    best_all_f1_d = all_f1_d
                    best_result = {'dev': [all_f1_d, test_f1_d, ign_f1_d, f1_d, auc_d, input_theta_dev]}
                    best_all_epoch = epoch
                    torch.save(model.state_dict(), self.best_model_path)
                    logging(f'FINAL (not best) model saved at {self.best_model_path}.')
                # Save best fine-tuned model
                else:
                    if all_f1_d > best_all_f1_d:
                        # Only do test eval on best dev models
                        logging('Test set evaluation:')
                        self.test_prefix = 'test'
                        self.load_data()
                        all_f1_t, test_f1_t, ign_f1_t, f1_t, auc_t, pr_x_t, pr_y_t, input_theta_test, all_learned_test, new_theta_test = self.eval(
                            model, save_name, epoch, input_theta=input_theta_dev, save_learn_order=False,
                            all_learned_uids=all_learned_dev)

                        best_all_f1_d = all_f1_d
                        best_result = {
                            'dev': {
                                'all_f1_d': all_f1_d,
                                'test_f1_d': test_f1_d,
                                'ign_f1_d': ign_f1_d,
                                'f1_d': f1_d,
                                'auc_d': auc_d,
                                'input_theta_dev': input_theta_dev
                            },
                            'test': {
                                'all_f1_t': all_f1_t,
                                'test_f1_t': test_f1_t,
                                'ign_f1_t': ign_f1_t,
                                'f1_t': f1_t,
                                'auc_t': auc_t,
                                'input_theta_test': input_theta_test
                            }
                        }
                        best_all_epoch = epoch
                        best_dev_theta = input_theta_dev
                        torch.save(model.state_dict(), self.best_model_path)
                        logging(f'Best model saved at {self.best_model_path}. Best theta value: {input_theta_dev}')
                logging('-' * 89)

            model.train()
            # END TRAINING LOOP

        logging(f'Best epoch {best_all_epoch}')
        logging('Best input_theta {:3.4f}'.format(best_dev_theta))
        logging(f'Best results: {best_result}')
        print("Finished training.")
        print('Logs saved: ', self.log_dir)

    def save_first_learned(self, first_learned, epoch):
        if self.config.training_data_type == 'annotated': return
        first_learned[epoch] = list(first_learned[epoch])
        int_epoch = int(epoch)

        # Full training eval aka Epoch-based:
        postfix = ''
        if 'train' in self.test_prefix and self.config.epoch_based_learning_order:
            postfix = '_whole'

        # Set correct save directory
        fname_fl = os.path.join(self.dir_first_learned_train, f'{self.test_prefix}_{int_epoch:03}{postfix}.json')

        # Save file
        print(f'Saving {fname_fl} with {len(first_learned[epoch])} UIDs...', end=' ')
        with open(fname_fl, "w") as wf:
            wf.write(json.dumps(first_learned) + "\n")
            print('complete.')

    def save_predictions(self, preds, epoch):
        if self.config.training_data_type == 'annotated': return
        list_preds = defaultdict(list)
        for tupl in preds[epoch]:
            list_preds[epoch].append(list(tupl))

        # Set correct save directory
        fname_preds = os.path.join(self.dir_dev_preds, f'{self.test_prefix}_final_preds.json')

        # Save file
        print(f'Saving {fname_preds}...')
        with open(fname_preds, "w") as wf:
            wf.write(json.dumps(list_preds, cls=NpEncoder) + "\n")

    def eval(self, model, save_name, epoch, output=False, input_theta=-1, save_learn_order=False, all_learned_uids=None,
             target_dir=None, output_embeddings=False, just_return_theta=False):
        rel_code2rel_id = json.load(open(f'{self.data_path}/rel2id.json'))
        rel_id2rel_code = {v: k for k, v in rel_code2rel_id.items()}  # invert rel2id dict
        rel_code2txt = json.load(open(f'{self.data_path}/../docred/rel_info.json'))

        all_rel_embeddings, all_rel_embeddings_labels = [], []
        learned_this_epoch = defaultdict(set)
        data_idx = 0
        eval_start_time = time.time()
        total_recall_ignore = 0

        test_result = []
        total_recall = 0

        predicted_as_zero = 0
        total_ins_num = 0

        def logging(s, print_=True, log_=True):
            if print_:
                print(s)
            if log_:
                with open(os.path.join(os.path.join(self.log_dir, save_name + '.log')), 'a+') as f_log:
                    f_log.write(s + '\n')

        acc_NA = Accuracy()
        acc_not_NA = Accuracy()
        acc_total = Accuracy()

        total_recall_per_class = defaultdict(int)  # macro recall
        for data in self.get_test_batch():
            with torch.no_grad():
                context_idxs = data['context_idxs']
                h_mapping = data['h_mapping']
                t_mapping = data['t_mapping']
                relation_label = data['relation_label']
                relation_uid = data['relation_uid']
                labels = data['labels']
                # label_uids = data['label_uids']
                labels_multi = data['label_multi']
                L_vertex = data['L_vertex']
                ht_pair_pos = data['ht_pair_pos']
                context_masks = data['context_masks']
                all_test_idxs = data['all_test_idxs']

                titles = data['titles']
                indexes = data['indexes']

                dis_h_2_t = ht_pair_pos + 10
                dis_t_2_h = -ht_pair_pos + 10

                predict_re = model(context_idxs, h_mapping, t_mapping, dis_h_2_t, dis_t_2_h, context_masks)

                if output_embeddings:
                    doc_rel_embeddings = []
                    doc_rel_embeddings_labels = []
                    for i in range(predict_re.shape[0]):
                        for j in range(predict_re.shape[1]):
                            label = relation_label[i][j].item()
                            if label == 0:
                                continue  # Rel = NA
                            embedding = predict_re[i][j]

                            # save representations --> predict_re
                            embedding = torch.squeeze(embedding.to(dtype=torch.half)).detach().cpu().numpy()
                            # save labels -->  relation_label
                            rel_code = rel_id2rel_code[label]
                            if rel_code == 'Na':
                                rel_txt = 'NA'
                            else:
                                rel_txt = rel_code2txt[rel_code]
                            # collect embeddings
                            doc_rel_embeddings.append(embedding)
                            doc_rel_embeddings_labels.append(rel_txt)
                    all_rel_embeddings.append(doc_rel_embeddings)
                    all_rel_embeddings_labels.append(doc_rel_embeddings_labels)
                    continue

                ##### START FIRST LEARNED
                if save_learn_order:
                    predict_arg_max = torch.argmax(predict_re, dim=-1)
                    predict_arg_max = predict_arg_max.data.cpu().numpy()
                    relation_label = relation_label.data.cpu().numpy()
                    for i in range(predict_arg_max.shape[0]):
                        for j in range(predict_arg_max.shape[1]):
                            label = relation_label[i][j]
                            uid = relation_uid[i][j].item()
                            if label < 0:
                                break
                            elif (label == 0):  # label is 'NA'
                                acc_NA.add(predict_arg_max[i][j] == label)
                                continue
                            else:  # All positive rels
                                acc_not_NA.add(predict_arg_max[i][j] == label)
                                if predict_arg_max[i][j] == label and uid not in all_learned_uids:
                                    all_learned_uids.add(uid)
                                    learned_this_epoch[epoch].add(uid)
                            acc_total.add(predict_arg_max[i][j] == label)
                ##### END FIRST LEARNED

            # Normal test stats:
            predict_re = torch.sigmoid(predict_re)
            predict_re = predict_re.data.cpu().numpy()
            for i in range(len(labels)):
                label = labels[i]
                label_multi = labels_multi[i]
                index = indexes[i]

                total_recall += len(label)
                for l in label.values():
                    if not l:
                        total_recall_ignore += 1

                L = L_vertex[i]
                test_idxs = all_test_idxs[i]
                j = 0

                # Marco label counter
                for h_t_r, in_train_bool in label.items():
                    h, t, r = h_t_r
                    total_recall_per_class[r] += 1

                for (h_idx, t_idx) in test_idxs:
                    r = np.argmax(predict_re[i, j])
                    predicted_as_zero += (r == 0)
                    total_ins_num += 1

                    for r in range(1, self.relation_num):
                        intrain = False
                        multi = -1
                        if (h_idx, t_idx, r) in label:
                            if label[(h_idx, t_idx, r)] == True:
                                intrain = True
                            multi = label_multi[(h_idx, t_idx)]
                        test_result.append(((h_idx, t_idx, r) in label, float(predict_re[i, j, r]), (intrain, multi),
                                            titles[i], self.id2rel[r], index, h_idx, t_idx, r))
                    j += 1
            data_idx += 1

            if data_idx % self.log_period == 0:
                print(
                    '| step {:3d} | time: {:5.2f}'.format(data_idx // self.log_period, (time.time() - eval_start_time)))
                eval_start_time = time.time()

        if output_embeddings:
            logging('SAVING output embeddings')
            np.save('dev_rel_embeddings.npy', all_rel_embeddings, allow_pickle=True)
            with open('dev_rel_embeddings_labels.tsv', 'w') as out:
                wr = csv.writer(out, delimiter='\t')
                wr.writerows(all_rel_embeddings_labels)

            return

        # Return set of learned UIDs
        if save_learn_order:
            logging(f'SAVING FIRST LEARNED DEV STATS: | epoch {epoch:2d} | '
                    f'NA acc: {acc_NA.get():4.2f} '
                    f'| not NA acc: {acc_not_NA.get():4.2f} '
                    f'| tot acc: {acc_total.get():4.2f} ')
            self.save_first_learned(learned_this_epoch, epoch)

        # If not saving first learned, continue eval like normal
        test_result.sort(key=lambda x: x[1], reverse=True)

        pr_x = []
        pr_y = []
        correct = 0

        w, w_orig = 0, 0

        if total_recall == 0:
            total_recall = 1  # for test

        # Memory management
        del data

        for i, item in enumerate(test_result):
            correct += item[0]

            pr_y.append(float(correct) / (i + 1))
            pr_x.append(float(correct) / total_recall)

            if item[1] > input_theta:
                w = i
        logging(f"Total Correct: {correct}")
        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        f1 = f1_arr.max()
        f1_pos = f1_arr.argmax()
        all_f1 = f1
        theta = test_result[f1_pos][1]  # softmax val for best f1

        if input_theta == -1:
            w = f1_pos
            input_theta = theta
        auc = sklearn.metrics.auc(x=pr_x, y=pr_y)
        if not self.is_test:
            new_theta = theta
            logging('Theta {:3.4f} | F1 {:3.4f} | AUC {:3.4f}'.format(theta, f1, auc))
        else:
            new_theta = None
            logging('ma_f1 {:3.4f} | input_theta {:3.4f} test_result F1 {:3.4f} | AUC {:3.4f}'.format(f1, input_theta,
                                                                                                      f1_arr[w], auc))
        test_f1 = f1_arr[w]

        if just_return_theta:
            logging(f'Returning from dev eval with new theta value, only.')
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, new_theta

        if output:
            output = [{'index': x[-4], 'h_idx': x[-3], 't_idx': x[-2], 'r_idx': x[-1], 'r': x[-5], 'title': x[-6]} for x
                      in test_result[:w + 1]]
            out_file = save_name + "_" + self.test_prefix + f'_{str(input_theta)}theta.json'
            if target_dir:  # save results in the original experiment folder
                out_file = os.path.join(target_dir, out_file)
            json.dump(output, open(out_file, "w"))

        pr_x = []
        pr_y = []
        correct = correct_in_train = 0
        w = 0

        pr_x_macro = defaultdict(list)
        pr_y_macro = defaultdict(list)
        correct_per_class = defaultdict(int)

        for i, item in enumerate(test_result):
            correct += item[0]
            if item[0] & item[2][0]:
                correct_in_train += 1
            if correct_in_train == correct:
                p = 0
            else:
                p = float(correct - correct_in_train) / (i + 1 - correct_in_train)
            pr_y.append(p)
            pr_x.append(float(correct) / total_recall)
            if item[1] > input_theta:
                w = i

                # Macro F1
                rel_idx = item[8]
                correct_per_class[rel_idx] += item[0]
                pr_y_macro[rel_idx].append(float(correct_per_class[rel_idx]) / (len(pr_y_macro[rel_idx]) + 1))
                pr_x_macro[rel_idx].append(float(correct_per_class[rel_idx]) / total_recall_per_class[rel_idx])

        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        ign_f1 = f1_arr.max()

        # Macro F1
        pr_y_macro_np = {}
        macro_f1_all = {}
        for rel_idx, y_list in pr_y_macro.items():
            pr_y_macro_np[rel_idx] = np.asarray(y_list, dtype='float32')
        for rel_idx, x_list in pr_x_macro.items():
            r = np.asarray(x_list, dtype='float32')[-1]
            p = pr_y_macro_np[rel_idx][-1]
            f1 = (2 * p * r / (p + r + 1e-20))
            macro_f1_all[rel_idx] = f1

        fl_macro_avg = 0
        fl_macro_avg_weighted = 0
        n_classes = len(macro_f1_all)
        n_instances = 0
        for rel_idx, f1_macro in macro_f1_all.items():
            fl_macro_avg += f1_macro
            fl_macro_avg_weighted += (f1_macro * total_recall_per_class[rel_idx])
            n_instances += total_recall_per_class[rel_idx]
        fl_macro_avg = fl_macro_avg / n_classes
        fl_macro_avg_weighted = fl_macro_avg_weighted / n_instances

        auc = sklearn.metrics.auc(x=pr_x, y=pr_y)
        logging(
            'Ignore ma_f1 {:3.4f} | input_theta {:3.4f} test_result F1 {:3.4f} | AUC {:3.4f} | all F1 {:3.4f} | macro f1 {:3.4f} | w_macro f1 {:3.4f}'.format(
                ign_f1, input_theta, f1_arr[w], auc, all_f1, fl_macro_avg, fl_macro_avg_weighted))

        return all_f1, test_f1, ign_f1, f1_arr[w], auc, pr_x, pr_y, input_theta, all_learned_uids, new_theta

    def get_confidence(self, model):
        for i in range(10):
            confidence_vals = {}
            self.test_prefix = f'train_distant_{i}'
            self.config.erica_file_num = i

            self.load_data()
            train_dataset = REDataset(self.train_prefix, self.data_path, self.h_t_limit, self.config)
            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=32,
                                          collate_fn=self.get_train_batch, num_workers=0)
            for batch in train_dataloader:
                with torch.no_grad():
                    data = {k: v.to(self.device) for k, v in batch.items()}
                    context_idxs = data['context_idxs']
                    h_mapping = data['h_mapping']
                    t_mapping = data['t_mapping']
                    # relation_label = data['relation_label']
                    relation_uid = data['relation_uid']
                    ht_pair_pos = data['ht_pair_pos']
                    context_masks = data['context_masks']
                    dis_h_2_t = ht_pair_pos + 10
                    dis_t_2_h = -ht_pair_pos + 10

                    predict_re = model(context_idxs, h_mapping, t_mapping, dis_h_2_t, dis_t_2_h, context_masks)

                    ##### START CONFIDENCE
                    softmax_vals = torch.nn.functional.softmax(predict_re, dim=2)
                    softmax_vals_at_arg_max, predict_arg_max = torch.max(softmax_vals, dim=-1)

                    # Detach
                    predict_arg_max = predict_arg_max.data.cpu().numpy()
                    softmax_vals_at_arg_max = softmax_vals_at_arg_max.data.cpu().numpy()
                    softmax_vals_at_arg_max_round = np.round(softmax_vals_at_arg_max, 5).astype(np.float16)

                    for i in range(predict_arg_max.shape[0]):
                        for j in range(predict_arg_max.shape[1]):
                            softmax_val = softmax_vals_at_arg_max_round[i][j]
                            uid = relation_uid[i][j].item()
                            if uid > 0 and uid in confidence_vals:
                                raise ValueError('UID has already been seen by model. This should not happen.')
                            confidence_vals[uid] = softmax_val  # save prediction tuple: (pred, label, uid)
                    ##### END CONFIDENCE

            # Save file
            if not os.path.exists(self.dir_first_learned):
                os.makedirs(self.dir_first_learned)

            fname_preds = os.path.join(self.dir_first_learned, f'{self.test_prefix}_confidence_dict.json')
            print(f'Saving {fname_preds} with {len(confidence_vals)} entries...')
            with open(fname_preds, "w") as wf:
                wf.write(json.dumps(confidence_vals, cls=NpEncoder) + "\n")

    def run_wcl(self, model_type, model_name_or_path, best_model_path):
        print(f'WCL METHOD: evaluating best model path --> {best_model_path}, model type: {model_type}')
        bert_model = MODEL_CLASSES[model_type].from_pretrained(model_name_or_path)
        model = REModel(config=self, bert_model=bert_model)
        # model.load_state_dict(torch.load(best_model_path, map_location=torch.device(self.device)))

        model.to(self.device)
        model.eval()
        self.get_confidence(model)

    def test(self, model_type, model_name_or_path, save_name, input_theta, best_model_path, target_dir):
        print(f'Evaluating {model_type}, {model_name_or_path}, best model path --> {best_model_path}')
        bert_model = MODEL_CLASSES[model_type].from_pretrained(model_name_or_path)
        model = REModel(config=self, bert_model=bert_model)

        model.load_state_dict(torch.load(best_model_path, map_location=torch.device(self.device)))
        model.to(self.device)
        model.eval()

        epoch = 0
        all_learned_uids = None

        for split in ['dev', 'test']:
            self.test_prefix = split
            output = (self.test_prefix == 'test')
            print('-' * 89)
            print(f'Beginning evaluation of {split} split, using {input_theta} theta.')
            print('-' * 89)
            self.load_data()
            just_return_theta = ('dev' in split)
            _, _, _, _, _, _, _, _, _, new_theta = self.eval(model, save_name, epoch, output=output,
                                                             input_theta=input_theta, target_dir=target_dir,
                                                             save_learn_order=self.config.error_analysis,
                                                             all_learned_uids=all_learned_uids,
                                                             just_return_theta=just_return_theta)
            if split == 'dev':
                input_theta = new_theta

        # Rename eval output dir
        print(f'Renamed {os.getcwd()} to {os.getcwd()}__{self.config.model_path_datetime} ')
        os.rename(os.getcwd(), os.getcwd() + f'__{self.config.model_path_datetime}')


def train_stage_1(config):
    con = Controller(config)
    con.train(config.model_type, config.model_name_or_path, config.trainer.save_name, config)
    return

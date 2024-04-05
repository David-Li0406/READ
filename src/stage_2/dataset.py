import os
import re
import sys
import json
import random

import math
import torch
import numpy as np
from os.path import join
from collections import Counter
from torch.utils import data
from src.stage_2.utils import EntityMarker
from collections import defaultdict
from tqdm import tqdm


DEFAULT_NEG_WEIGHT = 0.5
K = 5


class Scheduler:
    def __init__(self, order_dict) -> None:
        values = list(set(order_dict.values()))
        self.max_v = max(values)
        self.min_v = min(values)
        self.k = self.max_v - self.min_v

    def convert(self, value, method='linear'):
        if method == 'linear':
            return self._linear(value)
        else:
            raise NotImplementedError

    def _linear(self, value):
        if value >= K:
            value = self.max_v
        return 1 - (value - self.min_v) / self.k


def save_data(data, data_name):
    with open(data_name, 'w') as fout:
        json.dump(data, fout)


class CP_R_Dataset(data.Dataset):
    def __init__(self, config):
        self.config = config

        self.path = join(config.trainer.data_dir, 'erica_data')

        # if self.config.trainer.fine_grained:
        self.order_dict = json.load(open(join(self.path, 'order_dict_augmented.json')))
        self.scheduler = Scheduler(self.order_dict)

        self.rel2id = json.load(open(join(self.path, "rel2id.json")))
        self.id2rel = {v: k for k, v in self.rel2id.items()}

        self.P_info = json.load(open(join(self.path, 'P_info_tokenized_roberta.json'), 'r'))
        self.h_t_limit = 1000
        self.neg_multiple = 32
        self.relation_num = len(list(self.rel2id.keys()))
        self.max_length = config.trainer.max_length

        self.entityMarker = EntityMarker(config)
        if config.model_name_or_path == 'bert':
            self.idx2token = {v: k for k, v in self.entityMarker.tokenizer.vocab.items()}

        self.type2mask = ['[unused' + str(x) + ']' for x in range(1, 101)]
        self.start_token = ['[unused' + str(x) + ']' for x in range(101, 201)]
        self.end_token = ['[unused' + str(x) + ']' for x in range(201, 301)]

        self.epoch = 0

        self.__sample__()

    def get_data(self, ori_data, data_type):
        for i in tqdm(range(len(ori_data)), ncols=100, desc=f'getting {data_type} data...'):
            vertexSet = ori_data[i]['vertexSet']
            sent_id_to_vertex = {}

            if data_type == 'wiki_data':
                time = 0
                while (True):
                    if time > 10000:
                        print('!!!')
                    time += 1

                    hop_rel = random.choice(ori_data[i]['labels'])
                    hop_rel['r'] = self.id2rel[hop_rel['r']]
                    if hop_rel['r'] in self.P_info:
                        break

                query_h = hop_rel['h']
                query_t = hop_rel['t']
                query_r = hop_rel['r']

                query = []

                for ddd_word in self.P_info[query_r]:
                    query += ddd_word
                sent_id_q = ori_data[i]['vertexSet'][query_h][0]['sent_id']
                pos_q = ori_data[i]['vertexSet'][query_h][0]['pos']
                sent_q = ori_data[i]['sents'][sent_id_q]
                query += sent_q[pos_q[0]: pos_q[1]]
                query += [self.entityMarker.tokenizer.sep_token]

                ori_data[i]['sents'][0][0: 0] = iter(query)
                for x in range(len(ori_data[i]['vertexSet'])):
                    for y in range(len(ori_data[i]['vertexSet'][x])):
                        for z in range(len(ori_data[i]['vertexSet'][x][y]['pos'])):
                            ori_data[i]['vertexSet'][x][y]['pos'][z] += len(query)

                ori_data[i]['label_hop'] = hop_rel

                for jj in range(len(vertexSet)):
                    for k in range(len(vertexSet[jj])):
                        sent_id = int(vertexSet[jj][k]['sent_id'])
                        if sent_id not in sent_id_to_vertex:
                            sent_id_to_vertex[sent_id] = []
                        sent_id_to_vertex[sent_id].append([jj, k])


            elif data_type == 'doc_data':
                for jj in range(len(vertexSet)):
                    for k in range(len(vertexSet[jj])):
                        sent_id = int(vertexSet[jj][k]['sent_id'])
                        if sent_id not in sent_id_to_vertex:
                            sent_id_to_vertex[sent_id] = []
                        sent_id_to_vertex[sent_id].append([jj, k])
            else:
                assert False

            ori_data[i]['vertexSet'] = vertexSet

        return ori_data

    def check_data(self, data):

        checked = []

        for line in data:
            labels = line['labels']

            checked_labels = []
            for label in labels:
                if label['r'] in self.id2rel and self.id2rel[label['r']] in self.P_info:
                    checked_labels.append(label)

            if len(checked_labels) > 0:
                line['labels'] = checked_labels
                checked.append(line)

        print(f'We read {len(checked)} data lines..')
        return checked

    def __sample__(self):

        file_id = random.randint(0, 9)
        file_name = f'train_distant_{file_id}.json'
        print(f'Read data file {file_name}.')
        # self.config.trainer.dataset_name
        train_data = json.load(open(join(self.path, file_name), 'r'))

        train_data = self.check_data(train_data)
        random.shuffle(train_data)
        self.half_data_len = int(0.5 * len(train_data))

        self.doc_data = self.get_data(train_data[: self.half_data_len], 'doc_data')
        self.wiki_data = self.get_data(train_data[self.half_data_len: 2 * self.half_data_len], 'wiki_data')

        save_data(self.doc_data, 'doc_data.json')
        save_data(self.wiki_data, 'wiki_data.json')

        self.tokens = np.zeros((len(self.doc_data) + len(self.wiki_data), self.config.trainer.max_length), dtype=int)
        self.mask = np.zeros((len(self.doc_data) + len(self.wiki_data), self.config.trainer.max_length), dtype=int)
        self.bert_starts_ends = np.ones((len(self.doc_data) + len(self.wiki_data), self.config.trainer.max_length, 2),
                                        dtype=np.int64) * (self.config.trainer.max_length - 1)

        self.numberize(self.doc_data, 0)
        self.numberize(self.wiki_data, self.half_data_len)

        self.pos_pair = []
        scope = list(range(self.half_data_len))
        random.shuffle(scope)
        self.pos_pair = scope
        print("Positive pair's number is %d" % (len(self.pos_pair)))

    def numberize(self, data, offset):
        self.bad_instance_id = []
        wired_example_num = 0
        for i in tqdm(range(len(data)), ncols=100, desc='numberize...'):
            item = data[i]
            words = []
            for sent in item['sents']:
                words += sent
            idxs = []
            text = ""
            for word in words:
                if len(text) > 0:
                    text = text + " "
                idxs.append(len(text))
                text += word

            subwords = self.entityMarker.tokenizer.tokenize(text)

            char2subwords = []
            L = 0
            sub_idx = 0
            L_subwords = len(subwords)
            bad_flag = False
            while sub_idx < L_subwords:
                subword_list = []
                prev_sub_idx = sub_idx
                while sub_idx < L_subwords:
                    subword_list.append(subwords[sub_idx])
                    sub_idx += 1
                    subword = self.entityMarker.tokenizer.convert_tokens_to_string(subword_list)
                    sub_l = len(subword)
                    if subword == '</s>':
                        L += 1
                    if text[L:L + sub_l] == subword:
                        break

                if text[L:L + sub_l] != subword:
                    bad_flag = True
                    break

                assert (text[L:L + sub_l] == subword)
                if subword == '</s>':
                    char2subwords.extend([prev_sub_idx] * (sub_l + 1))
                else:
                    char2subwords.extend([prev_sub_idx] * sub_l)

                L += len(subword)

            if len(text) != len(char2subwords) or bad_flag:
                wired_example_num += 1
                self.bad_instance_id.append(i + offset)
                continue
                # text = text[:len(char2subwords)]

            assert (len(text) == len(char2subwords))
            tokens = [self.entityMarker.tokenizer.cls_token] + subwords[: 512 - 2] + [
                self.entityMarker.tokenizer.sep_token]

            L_ori = len(tokens)
            tokens = self.entityMarker.tokenizer.convert_tokens_to_ids(tokens)

            pad_len = 512 - len(tokens)
            mask = [1] * len(tokens) + [0] * pad_len
            tokens = tokens + [1] * pad_len

            self.tokens[i + offset] = tokens
            self.mask[i + offset] = mask

            for j in range(len(words)):
                if j >= 512:  # TODO: added this to prevent out of index error. Word sequences should be capped at 512!
                    break
                idx = char2subwords[idxs[j]] + 1
                idx = min(idx, 512 - 1)

                x = idxs[j] + len(words[j])
                if x == len(char2subwords):
                    idx2 = L_ori
                else:
                    idx2 = char2subwords[x] + 1
                    idx2 = min(idx2, 512 - 1)
                self.bert_starts_ends[i + offset, j, 0] = idx
                self.bert_starts_ends[i + offset, j, 1] = idx2

        print('wired_example_num: ' + str(wired_example_num))

    def __len__(self):
        return len(self.pos_pair)

    def __getitem__(self, index):
        while index in self.bad_instance_id or index + self.half_data_len in self.bad_instance_id:
            print('reselecting another instance')
            index = random.choice(list(range(500)))
        bag_idx = self.pos_pair[index]

        ids = self.tokens[bag_idx]
        mask = self.mask[bag_idx]
        bert_starts_ends = self.bert_starts_ends[bag_idx]
        item = self.doc_data[bag_idx]

        ids_w = self.tokens[bag_idx + self.half_data_len]
        mask_w = self.mask[bag_idx + self.half_data_len]
        bert_starts_ends_w = self.bert_starts_ends[bag_idx + self.half_data_len]
        item_w = self.wiki_data[bag_idx]

        return (ids, mask, bert_starts_ends, item), (ids_w, mask_w, bert_starts_ends_w, item_w)

    def get_doc_batch(self, batch):
        batch_size = len(batch)

        max_length = self.max_length
        h_t_limit = self.h_t_limit
        context_idxs = torch.LongTensor(batch_size, max_length).zero_()
        h_mapping = torch.Tensor(batch_size, h_t_limit, max_length).zero_()
        t_mapping = torch.Tensor(batch_size, h_t_limit, max_length).zero_()
        mlm_mask = torch.Tensor(batch_size, max_length).zero_()
        context_masks = torch.LongTensor(batch_size, max_length).zero_()

        rel_list = {}
        rel_list_none = []
        max_h_t_cnt = -1
        idx2pair = {}
        uid2weights = {}
        for i in range(len(batch)):
            # import pdb; pdb.set_trace()
            item = batch[i]
            context_idxs[i].copy_(torch.from_numpy(item[0]))
            context_masks[i].copy_(torch.from_numpy(item[1]))

            item = batch[i][3]
            starts_pos = batch[i][2][:, 0]
            ends_pos = batch[i][2][:, 1]
            labels = item['labels']

            idx2label = defaultdict(list)

            for label in labels:
                idx2label[(label['h'], label['t'])].append((label['r'], str(label['uid'])))

            train_triple = list(idx2label.keys())
            j = 0
            for (h_idx, t_idx) in train_triple:
                if j == self.h_t_limit:
                    break

                hlist = item['vertexSet'][h_idx]
                tlist = item['vertexSet'][t_idx]

                hlist = [(starts_pos[h['pos'][0]], ends_pos[h['pos'][1] - 1]) for h in hlist if
                         h['pos'][1] < 511 and ends_pos[h['pos'][1] - 1] < 511]
                tlist = [(starts_pos[t['pos'][0]], ends_pos[t['pos'][1] - 1]) for t in tlist if
                         t['pos'][1] < 511 and ends_pos[t['pos'][1] - 1] < 511]

                if len(hlist) == 0 or len(tlist) == 0:
                    continue

                for h in hlist:
                    h_mapping[i, j, h[0]:h[1]] = 1.0 / len(hlist) / (h[1] - h[0])
                    mlm_mask[i, h[0]: h[1]] = 1

                for t in tlist:
                    t_mapping[i, j, t[0]:t[1]] = 1.0 / len(tlist) / (t[1] - t[0])
                    mlm_mask[i, t[0]: t[1]] = 1

                label = idx2label[(h_idx, t_idx)]

                for (r_idx, uid) in label:

                    if uid not in self.order_dict:
                        continue
                    if r_idx not in rel_list:
                        rel_list[r_idx] = []
                    rel_list[r_idx].append([i, j, uid])
                    uid2weights[uid] = self.scheduler.convert(self.order_dict[uid])  # self.order_dict[uid]#

                idx2pair[(i, j)] = [h_idx, t_idx]

                j += 1

            if not self.config.trainer.add_none:
                max_h_t_cnt = max(max_h_t_cnt, len(train_triple))
            else:
                lower_bound = min(len(item['na_triple']), self.neg_multiple)
                sel_ins = random.sample(item['na_triple'], lower_bound)

                for (h_idx, t_idx) in sel_ins:
                    if j == self.h_t_limit:
                        break
                    hlist = item['vertexSet'][h_idx]
                    tlist = item['vertexSet'][t_idx]

                    hlist = [(starts_pos[h['pos'][0]], ends_pos[h['pos'][1] - 1]) for h in hlist if
                             h['pos'][1] < 511 and ends_pos[h['pos'][1] - 1] < 511]
                    tlist = [(starts_pos[t['pos'][0]], ends_pos[t['pos'][1] - 1]) for t in tlist if
                             t['pos'][1] < 511 and ends_pos[t['pos'][1] - 1] < 511]

                    hlist = [x for x in hlist if x[0] < x[1]]
                    tlist = [x for x in tlist if x[0] < x[1]]

                    if len(hlist) == 0 or len(tlist) == 0:
                        continue

                    for h in hlist:
                        h_mapping[i, j, h[0]:h[1]] = 1.0 / len(hlist) / (h[1] - h[0])
                        mlm_mask[i, h[0]: h[1]] = 1

                    for t in tlist:
                        t_mapping[i, j, t[0]:t[1]] = 1.0 / len(tlist) / (t[1] - t[0])
                        mlm_mask[i, t[0]: t[1]] = 1

                    rel_list_none.append([i, j])
                    idx2pair[(i, j)] = [h_idx, t_idx]

                    j += 1

                max_h_t_cnt = max(max_h_t_cnt, len(train_triple) + lower_bound)

        for k in rel_list:
            random.shuffle(rel_list[k])
        random.shuffle(rel_list_none)
        rel_sum_not_none = sum([len(rel_list[k]) for k in rel_list])
        total_rel_sum = rel_sum_not_none + len(rel_list_none)

        relation_label = torch.LongTensor(total_rel_sum).zero_()
        relation_label_idx = torch.LongTensor(total_rel_sum, 2).zero_()
        relation_weights = torch.FloatTensor(total_rel_sum).zero_()
        pos_num = 0
        jj = -1 - len(rel_list_none)

        for k in rel_list:
            for j in range(len(rel_list[k])):
                if len(rel_list[k]) % 2 == 1 and j == len(rel_list[k]) - 1:
                    break
                relation_label[pos_num] = k
                relation_label_idx[pos_num] = torch.LongTensor(
                    rel_list[k][j][:2])  # :2 the first two elements are index
                relation_weights[pos_num] = uid2weights[rel_list[k][j][2]]  # the third element is uid
                pos_num += 1
            if len(rel_list[k]) % 2 == 1:
                relation_label[jj] = k
                relation_label_idx[jj] = torch.LongTensor(rel_list[k][-1][:2])
                relation_weights[jj] = uid2weights[rel_list[k][-1][2]]
                jj -= 1

        for j in range(len(rel_list_none)):
            relation_label[rel_sum_not_none + j] = 0
            relation_label_idx[rel_sum_not_none + j] = torch.LongTensor(rel_list_none[j])
            relation_weights[rel_sum_not_none + j] = DEFAULT_NEG_WEIGHT

        rel_mask_pos = torch.LongTensor(total_rel_sum, total_rel_sum).zero_()
        rel_mask_neg = torch.LongTensor(total_rel_sum, total_rel_sum).zero_()
        for i in range(total_rel_sum):
            if i >= pos_num:
                break
            neg = []
            pos = []
            for j in range(total_rel_sum):
                idx_1 = relation_label_idx[i].numpy().tolist()
                idx_2 = relation_label_idx[j].numpy().tolist()
                pair_idx_1 = idx2pair[tuple(idx_1)]
                pair_idx_2 = idx2pair[tuple(idx_2)]
                if idx_1[0] == idx_2[0]:
                    if pair_idx_1[0] == pair_idx_2[0] or pair_idx_1[1] == pair_idx_2[1]:
                        continue
                if relation_label[i] != relation_label[j] and idx_1 != idx_2:
                    neg.append(j)
                if relation_label[i] == relation_label[j] and idx_1 != idx_2:
                    if i % 2 == 0 and j == i + 1:
                        pos.append(j)
                    elif i % 2 == 1 and j == i - 1:
                        pos.append(j)

            if len(neg) > self.config.trainer.neg_sample_num:
                neg = random.sample(neg, self.config.trainer.neg_sample_num)
            for j in neg:
                rel_mask_neg[i, j] = 1
            for j in pos:
                rel_mask_pos[i, j] = 1

        return {'context_idxs': context_idxs,
                'h_mapping': h_mapping[:, :max_h_t_cnt, :].contiguous(),
                't_mapping': t_mapping[:, :max_h_t_cnt, :].contiguous(),
                'relation_label': relation_label.contiguous(),
                'relation_label_idx': relation_label_idx.contiguous(),
                'relation_weights': relation_weights.contiguous(),
                'context_masks': context_masks,
                'rel_mask_pos': rel_mask_pos,
                'rel_mask_neg': rel_mask_neg,
                'pos_num': torch.tensor([pos_num]).cuda(),
                'mlm_mask': mlm_mask,
                }

    def get_wiki_batch(self, batch):
        batch_size = len(batch)
        if batch_size == 0:
            return {}
        max_length = self.max_length
        h_t_limit = self.h_t_limit
        context_idxs = torch.LongTensor(batch_size, max_length).zero_()

        h_mapping = torch.Tensor(batch_size, h_t_limit, max_length).zero_()
        query_mapping = torch.Tensor(batch_size, max_length).zero_()
        mlm_mask = torch.Tensor(batch_size, max_length).zero_()

        context_masks = torch.LongTensor(batch_size, max_length).zero_()
        start_positions = torch.LongTensor(batch_size).zero_()
        end_positions = torch.LongTensor(batch_size).zero_()

        rel_mask_pos = torch.LongTensor(len(batch), self.h_t_limit).zero_()
        rel_mask_neg = torch.LongTensor(len(batch), self.h_t_limit).zero_()

        j = 0

        # import pdb; pdb.set_trace()
        relation_label_idx = []
        relation_label = []
        for i in range(len(batch)):
            item = batch[i]
            context_idxs[i].copy_(torch.from_numpy(item[0]))
            context_masks[i].copy_(torch.from_numpy(item[1]))

            item = batch[i][3]

            starts_pos = batch[i][2][:, 0]
            ends_pos = batch[i][2][:, 1]

            hlist = item['vertexSet'][item['label_hop']['h']]
            hlist = [(starts_pos[h['pos'][0]], ends_pos[h['pos'][1] - 1]) for h in hlist if
                     h['pos'][1] < 511 and ends_pos[h['pos'][1] - 1] < 511]
            hlist = [x for x in hlist if x[0] < x[1]]
            if len(hlist) == 0:
                continue
            for h in hlist:
                query_mapping[i, h[0]:h[1]] = 1.0 / len(hlist) / (h[1] - h[0])
                mlm_mask[i, h[0]: h[1]] = 1

            flag = 0
            for vertex_idx, vertex in enumerate(item['vertexSet']):
                if vertex_idx == item['label_hop']['h']:
                    continue
                hlist = item['vertexSet'][vertex_idx]
                hlist = [(starts_pos[h['pos'][0]], ends_pos[h['pos'][1] - 1]) for h in hlist if
                         h['pos'][1] < 511 and ends_pos[h['pos'][1] - 1] < 511]
                hlist = [x for x in hlist if x[0] < x[1]]
                if len(hlist) == 0:
                    continue

                if vertex_idx == item['label_hop']['t']:
                    rel_mask_pos[i, j] = 1.0
                    rel_mask_neg[i, j] = 1.0
                    answer = random.choice(hlist)
                    start_positions[i] = answer[0].tolist()
                    end_positions[i] = answer[1].tolist() - 1
                    flag = 1
                elif j == self.h_t_limit:
                    continue
                else:
                    rel_mask_neg[i, j] = 1.0
                    flag = 0

                for h in hlist:
                    h_mapping[i, j, h[0]:h[1]] = 1.0 / len(hlist) / (h[1] - h[0])
                    mlm_mask[i, h[0]: h[1]] = 1

                relation_label_idx.append([i, j])
                j += 1

        relation_label_idx = torch.LongTensor(relation_label_idx)
        return {'context_idxs': context_idxs,
                'h_mapping': h_mapping[:, : j, :].contiguous(),
                'query_mapping': query_mapping.contiguous(),
                'context_masks': context_masks,
                'rel_mask_pos': rel_mask_pos[:, :j],
                'rel_mask_neg': rel_mask_neg[:, :j],
                'relation_label_idx': relation_label_idx,
                'start_positions': start_positions,
                'end_positions': end_positions,
                'mlm_mask': mlm_mask,
                }

    def get_train_batch(self, batch):
        batch_doc = self.get_doc_batch([b[0] for b in batch])
        batch_wiki = self.get_wiki_batch([b[1] for b in batch])
        return [batch_doc, batch_wiki]

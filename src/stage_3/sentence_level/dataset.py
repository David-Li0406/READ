import os
import re
import ast
import sys
import json
import pdb
import random
import torch
import numpy as np
from torch.utils import data
from src.stage_3.sentence_level.utils import EntityMarker


class REDataset(data.Dataset):
    """
    Data loader for semeval, tacred
    """

    def __init__(self, path, mode, config):
        self.mode = mode
        data = []
        with open(os.path.join(path, mode)) as f:
            all_lines = f.readlines()
            for line in all_lines:
                ins = json.loads(line)
                data.append(ins)
        assert type(data[0]) is dict  # Make sure imported data is in the form: 1 dictionary per relation instance.

        entityMarker = EntityMarker(config)
        tot_instance = len(data)

        # load rel2id and type2id
        if os.path.exists(os.path.join(path, "rel2id.json")):
            rel2id = json.load(open(os.path.join(path, "rel2id.json")))
        else:
            raise Exception("Error: There is no `rel2id.json` in " + path + ".")

        print("pre process " + mode)
        # pre process data
        self.input_ids = np.zeros((tot_instance, config.trainer.max_length), dtype=int)
        self.mask = np.zeros((tot_instance, config.trainer.max_length), dtype=int)
        self.h_pos = np.zeros((tot_instance), dtype=int)
        self.t_pos = np.zeros((tot_instance), dtype=int)
        self.h_pos_l = np.zeros((tot_instance), dtype=int)
        self.t_pos_l = np.zeros((tot_instance), dtype=int)
        self.label = np.zeros((tot_instance), dtype=int)
        # if 'train' in self.mode:
        #     self.entity_clean_prob = np.zeros((tot_instance), dtype=float)
        #     self.context_clean_prob = np.zeros((tot_instance), dtype=float)

        for i, ins in enumerate(data):
            self.label[i] = rel2id[ins["relation"]]
            # tokenize
            if config.trainer.mode == "CM":
                ids, ph, pt, ph_l, pt_l = entityMarker.tokenize(data[i]["token"], data[i]['h']['pos'],
                                                                data[i]['t']['pos'])
            else:
                raise Exception("No such mode! Please make sure that `mode` takes the value in {CM,OC,CT,OM,OT}")

            length = min(len(ids), config.trainer.max_length)
            self.input_ids[i][0:length] = ids[0:length]
            self.mask[i][0:length] = 1
            self.h_pos[i] = min(ph, config.trainer.max_length - 1)
            self.t_pos[i] = min(pt, config.trainer.max_length - 1)
            self.h_pos_l[i] = min(ph_l, config.trainer.max_length)
            self.t_pos_l[i] = min(pt_l, config.trainer.max_length)
            # if 'train' in self.mode:
            #     self.entity_clean_prob[i] = data[i]["entity_clean_prob"]
            #     self.context_clean_prob[i] = data[i]["context_clean_prob"]
        print(
            f"Ratio of sentences in which tokenizer can't find head/tail entity is {entityMarker.err}/{entityMarker.n_total}, or {(entityMarker.err / entityMarker.n_total) * 100}%")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        input_ids = self.input_ids[index]
        mask = self.mask[index]
        h_pos = self.h_pos[index]
        t_pos = self.t_pos[index]
        h_pos_l = self.h_pos_l[index]
        t_pos_l = self.t_pos_l[index]
        label = self.label[index]
        # if 'train' in self.mode:
        #     entity_clean_prob = self.entity_clean_prob[index]
        #     context_clean_prob = self.context_clean_prob[index]
        #     return input_ids, mask, h_pos, t_pos, label, index, h_pos_l, t_pos_l, entity_clean_prob, context_clean_prob

        return input_ids, mask, h_pos, t_pos, label, index, h_pos_l, t_pos_l

import pdb
import torch
import torch.nn as nn
from src.transformers import BertModel, RobertaModel


class REModel(nn.Module):
    """
    Relation extraction model
    """

    def __init__(self, config, weight=None):
        super(REModel, self).__init__()
        self.config = config
        self.training = True

        if weight is None:
            self.loss = nn.CrossEntropyLoss()
        else:
            print("CrossEntropy Loss has weight!")
            self.loss = nn.CrossEntropyLoss(weight=weight)

        scale = 2 if config.trainer.entity_marker else 1
        self.rel_fc = nn.Linear(config.trainer.hidden_size * scale, config.trainer.rel_num)
        if 'bert-base-uncased' == config.model_name_or_path:
            self.bert = BertModel.from_pretrained(config.model_name_or_path)
            if config.pretrained_model_path != "None":
                print("********* load from ckpt/" + config.pretrained_model_path + " ***********")
                ckpt = torch.load(config.pretrained_model_path)
                self.bert.load_state_dict(ckpt["bert-base"])
            else:
                print("*******No ckpt to load, Let's use bert base!*******")
        elif 'roberta-base' == config.model_name_or_path:
            self.bert = RobertaModel.from_pretrained(config.model_name_or_path,
                                                     hidden_dropout_prob=config.trainer.dropout)
            if config.pretrained_model_path != "None":
                print("********* load from ckpt/" + config.pretrained_model_path + " ***********")
                ckpt = torch.load(config.pretrained_model_path)
                self.bert.load_state_dict(ckpt["bert-base"])
            else:
                print("*******No ckpt to load, Let's use bert base!*******")

    def forward(self, input_ids, mask, h_pos, t_pos, label, h_pos_l, t_pos_l, inputs_embeds=None):
        # bert encode
        if inputs_embeds is not None:
            outputs = self.bert(attention_mask=mask, inputs_embeds=inputs_embeds)
        else:
            outputs = self.bert(input_ids=input_ids, attention_mask=mask)

        # entity marker
        if self.config.trainer.entity_marker:
            indice = torch.arange(input_ids.size()[0])

            h_state = []
            t_state = []
            for i in range(input_ids.size()[0]):
                h_state.append(torch.mean(outputs[0][i, h_pos[i]: h_pos_l[i]], dim=0))
                t_state.append(torch.mean(outputs[0][i, t_pos[i]: t_pos_l[i]], dim=0))
            h_state = torch.stack(h_state, dim=0)
            t_state = torch.stack(t_state, dim=0)

            state = torch.cat((h_state, t_state), 1)  # (batch_size, hidden_size*2)
        else:
            # [CLS]
            state = outputs[0][:, 0, :]  # (batch_size, hidden_size)

        # linear map
        logits = self.rel_fc(state)  # (batch_size, rel_num)
        _, output = torch.max(logits, 1)

        if self.training:
            loss = self.loss(logits, label)
            return loss, output
        else:
            return logits, output

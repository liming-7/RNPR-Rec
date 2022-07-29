# -*- coding: utf-8 -*-
# @Time    : 17/11/2021 16:12
# @Author  : Ming

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import math

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class rankusr_model_explore_aware_model(nn.Module):

    def __init__(self, config):
        super(rankusr_model_explore_aware_model, self).__init__()
        self.cat_num = config['cat_num']
        self.emb_dim = config['embed_dim']
        self.freq_max_len = config['freq_max_len']
        self.emb_max_len = config['emb_max_len']
        self.dropout_rate = config['dropout_rate']
        self.device = config['device']
        self.freq_signal = config['freq_signal']
        self.avg_type = config['avg_type']

        self.bce_loss_layer = torch.nn.BCELoss(reduction='mean')
        self.bce_loss_layer_exp = torch.nn.BCELoss(reduction='mean')
        self.bce_logits_loss_layer = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.bce_logits_loss_layer_exp = torch.nn.BCEWithLogitsLoss(reduction='mean')

        self.cat_emb = nn.Embedding(self.cat_num+1, 12, padding_idx=self.cat_num) # controls the weight of different scources freq

        # similarity
        self.pos_global_sim_weight = nn.Embedding(1, self.emb_max_len)
        nn.init.constant_(self.pos_global_sim_weight.weight, 1.0)
        self.pos_cat_sim_weight = nn.Embedding(self.cat_num, self.emb_max_len)
        nn.init.constant_(self.pos_cat_sim_weight.weight, 1.0)
        self.sim_mix_emb = nn.Embedding(self.cat_num, 2) # whether this should be category specific?
        nn.init.constant_(self.sim_mix_emb.weight, 0.5)

        if self.freq_signal == 1:
            # frequency part
            self.incat_rep_pos_weight = nn.Embedding(self.cat_num, self.freq_max_len)
            nn.init.constant_(self.incat_rep_pos_weight.weight, 0.1)
            self.global_rep_pos_weight = nn.Embedding(1, self.freq_max_len)
            nn.init.constant_(self.global_rep_pos_weight.weight, 0.1)
            self.incat_exp_pos_weight = nn.Embedding(self.cat_num, self.freq_max_len)
            nn.init.constant_(self.incat_exp_pos_weight.weight, 0.1)
            self.global_exp_pos_weight = nn.Embedding(1, self.freq_max_len)
            nn.init.constant_(self.global_exp_pos_weight.weight, 0.1)

            self.freq_cat_emb = nn.Embedding(self.cat_num + 1, 12, padding_idx=self.cat_num)
            self.freq_linear_hidden = nn.Linear(16, 32)
            self.freq_output = nn.Linear(32, 1)
        else:
            self.linear_hidden = nn.Linear(14, 32)
            self.output = nn.Linear(32, 1)

    def forward(self, item, cat, sim_seq, rep_seq, expl_seq):
        '''
        input
        '''
        # similarity
        cat_sim_weight = self.pos_cat_sim_weight(cat)
        global_sim_weight = self.pos_global_sim_weight(torch.zeros_like(cat))
        cat_emb_specific = self.cat_emb(cat)

        cat_sim_score = torch.multiply(cat_sim_weight, sim_seq)
        cat_sim_score = torch.sum(cat_sim_score, dim=1).unsqueeze(1)
        global_sim_score = torch.multiply(global_sim_weight, sim_seq)
        global_sim_score = torch.sum(global_sim_score, dim=1).unsqueeze(1)

        sim_score = torch.cat([cat_sim_score, global_sim_score], dim=1)
        sim_mix_weight = self.sim_mix_emb(cat)
        joint_sim_score = torch.multiply(sim_score, sim_mix_weight)
        joint_sim_score = torch.sum(joint_sim_score, dim=1).unsqueeze(1)

        # frequency
        if self.freq_signal == 1:
            incat_rep_weight = self.incat_rep_pos_weight(cat)
            incat_exp_weight = self.incat_exp_pos_weight(cat)
            global_rep_weight = self.global_rep_pos_weight(torch.zeros_like(cat))
            global_exp_weight = self.global_exp_pos_weight(torch.zeros_like(cat))
            cat_rep_score = torch.multiply(rep_seq, incat_rep_weight)
            cat_rep_score = torch.sum(cat_rep_score, dim=1).unsqueeze(1)
            global_rep_score = torch.multiply(rep_seq, global_rep_weight)
            global_rep_score = torch.sum(global_rep_score, dim=1).unsqueeze(1)

            incat_exp_score = torch.multiply(expl_seq, incat_exp_weight)
            incat_exp_score = torch.sum(incat_exp_score, dim=1).unsqueeze(1)
            global_exp_score = torch.multiply(expl_seq, global_exp_weight)
            global_exp_score = torch.sum(global_exp_score, dim=1).unsqueeze(1)
            freq_cat_emb = self.freq_cat_emb(cat)
            freq_score = torch.cat([cat_rep_score, global_rep_score, incat_exp_score, global_exp_score, freq_cat_emb], dim=1)
            freq_score = self.freq_linear_hidden(freq_score)
            joint_freq_score = self.freq_output(freq_score) # exp score
            exp_prob = nn.Sigmoid()(joint_freq_score)
            final_score = exp_prob*joint_sim_score
        else:
            final_score = joint_sim_score

        return final_score

    def calculate_loss(self, pos_item, pos_cat, pos_sim_seq, pos_rep_seq, pos_expl_seq,
                       neg_item, neg_cat, neg_sim_seq, neg_rep_seq, neg_expl_seq):
        # pos cat items just for simplicity
        # print(pos_seq)
        pos_score = self.forward(pos_item, pos_cat, pos_sim_seq, pos_rep_seq, pos_expl_seq) # batch, cat_num
        neg_score = self.forward(neg_item, neg_cat, neg_sim_seq, neg_rep_seq, neg_expl_seq) # batch, cat_num
        pos_score = pos_score.squeeze()
        neg_score = neg_score.squeeze()
        diff_score = pos_score - neg_score
        label = torch.ones_like(diff_score, dtype=torch.float32).to(self.device)
        # print(nn.Sigmoid()(diff_score))
        loss = self.bce_logits_loss_layer(diff_score, label)
        return loss

    def calculate_performance(self, cat_seq, pos_seq, freq_vector, truth_vector):
        output = self.forward(cat_seq, pos_seq, freq_vector)
        prob_output = torch.sigmoid(output)
        prob_output[prob_output > 0.5] = 1.0
        prob_output[prob_output <= 0.5] = 0.0
        return 0

    def prob_prediction(self, item, cat, sim_seq, rep_seq, expl_seq):
        score = self.forward(item, cat, sim_seq, rep_seq, expl_seq)
        return score

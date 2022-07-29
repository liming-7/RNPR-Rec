# -*- coding: utf-8 -*-
# @Time    : 17/11/2021 16:12
# @Author  : Ming

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import math


class candidate_filter_model(nn.Module):

    def __init__(self, config):
        super(candidate_filter_model, self).__init__()
        self.cat_num = config['cat_num']
        self.freq_max_len = config['freq_max_len']
        self.dropout_rate = config['dropout_rate']
        self.device = config['device']
        self.model_version = config['model_version']


        self.bce_loss_layer = torch.nn.BCELoss(reduction='mean')
        self.bce_logits_loss_layer = torch.nn.BCEWithLogitsLoss(reduction='mean') #, pos_weight=torch.tensor([5])

        self.overall_pos_emb = nn.Embedding(1, self.freq_max_len) # overall which allows colaborative filtering among categories.
        nn.init.constant_(self.overall_pos_emb.weight, 1.0)

        self.cat_pos_weight = nn.Embedding(self.cat_num, self.freq_max_len)
        nn.init.constant_(self.cat_pos_weight.weight, 1.0)

        self.cat_feat_emb = nn.Embedding(self.cat_num, 14)

        self.linear_hidden = nn.Linear(16, 32)
        self.linear_output = nn.Linear(32, 1)

        # self.cat_way_weight = nn.Embedding(self.cat_num, 2)
        # nn.init.constant_(self.cat_pos_weight.weight, 1.0)

    def forward(self, cat, freq_vec_seq):
        cat_pos_weight = self.cat_pos_weight(cat)
        cat_rep_score = torch.multiply(freq_vec_seq, cat_pos_weight)
        cat_rep_score = torch.sum(cat_rep_score, dim=1).unsqueeze(1)

        overall_v = torch.zeros_like(cat)
        overall_pos_weight = self.overall_pos_emb(overall_v)
        overall_rep_score = torch.multiply(freq_vec_seq, overall_pos_weight)
        overall_rep_score = torch.sum(overall_rep_score, dim=1).unsqueeze(1)
        if self.model_version == 'v1':
            cat_feat = self.cat_feat_emb(cat)
            feat = torch.cat([cat_rep_score, overall_rep_score, cat_feat], dim=1)
            hidden = self.linear_hidden(feat)
            output = self.linear_output(hidden)
        elif self.model_version == 'v2':
            cat_way_weight = self.cat_way_weight(cat)
            feat = torch.cat([cat_rep_score, overall_rep_score], dim=1)
            output = torch.multiply(feat, cat_way_weight)
            output = torch.sum(output, dim=1)

        return output

    def calculate_loss_balanced(self, cat, pos_freq_vec_seq, neg_freq_vec_seq):
        pos_pred_score = self.forward(cat, pos_freq_vec_seq).squeeze()
        pos_label = torch.ones_like(pos_pred_score)
        neg_pred_score = self.forward(cat, neg_freq_vec_seq).squeeze()
        neg_label = torch.zeros_like(neg_pred_score)

        pred_score = torch.cat([pos_pred_score, neg_pred_score], axis=0)
        label = torch.cat([pos_label, neg_label], axis=0)
        loss = self.bce_logits_loss_layer(pred_score, label)
        return loss

    def calculate_loss(self, cat, freq_vec_seq, label):
        pred_score  = self.forward(cat, freq_vec_seq).squeeze()
        label = label.squeeze()
        loss = self.bce_logits_loss_layer(pred_score, label)
        return loss

    def calculate_performance(self, cat_seq, pos_seq, freq_vector, truth_vector):
        output = self.forward(cat_seq, pos_seq, freq_vector)
        prob_output = torch.sigmoid(output)
        prob_output[prob_output > 0.5] = 1.0
        prob_output[prob_output <= 0.5] = 0.0
        return 0

    def prob_prediction(self, cat, freq_vec_seq):
        output = self.forward(cat, freq_vec_seq)
        prob_output = nn.Sigmoid()(output).squeeze()
        return prob_output

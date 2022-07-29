# -*- coding: utf-8 -*-
# @Time    : 09/12/2021 12:41
# @Author  : Ming

import pickle
import os
import json
import numpy as np
import utils
import metrics
import torch

dataset = 'dunnhumby'
repeat_flag = False
if dataset == 'dunnhumby':
    min_new_usr = 10 # dunnhumby 10, instacart 50
if dataset == 'instacart':
    min_new_usr = 50
next_n = 5
epoch = 70
item_path = f'../data/{dataset}_temp/item_user_label{next_n}.pkl'
item_usr_label, item_set = utils.get_itemset(item_path, min_new_usr)
pred_path = f'pred/{dataset}/cand_1/exp_signal0/test_epoch_10.pkl'
with open(pred_path, 'rb') as f:
    iu_score_dict = pickle.load(f)

# load fold info
with open(f'../data/{dataset}_temp/foldk.pkl', 'rb') as f:
    foldk = pickle.load(f)

usr_seq_b_path = f'../data/{dataset}_temp/basket_seq_filter_all_30.pkl'
with open(usr_seq_b_path, 'rb') as f:
    usr_seq_b = pickle.load(f)
usr_item_set = dict()
for usr, bask_seq in usr_seq_b.items():
    usr_item_set[usr] = set([item for bask in bask_seq[:-10] for item in bask])

print(len(iu_score_dict.keys()))
# rank top-k users for item
fold_performance = dict()
print('Evaluating .......')
for fold in range(5):
    item_test_set = foldk[fold]['test']
    performance = dict()
    for topk in [10, 20, 50, 100, 200, 500, 1000]:
        perf_topk = dict()
        recall_list = []
        ndcg_list = []
        ht_list = []
        correct_list = []
        ctr_list = []
        for item in item_test_set:
            # print(item)

            if torch.tensor(item) not in iu_score_dict.keys():
                continue
            if repeat_flag:
                pred_usr = utils.get_pred_user(iu_score_dict[torch.tensor(item)], topk, t_item=item) #rep
                truth_usr = item_usr_label[item][0] + item_usr_label[item][1]
            else:
                pred_usr = utils.get_pred_user(iu_score_dict[torch.tensor(item)], topk, t_item=item, repeat_usr=usr_item_set) #explore
                truth_usr = item_usr_label[item][0]
            recall_list.append(metrics.get_Recall(truth_usr, pred_usr, topk))
            ndcg_list.append(metrics.get_NDCG(truth_usr, pred_usr, topk))
            ht_list.append(metrics.get_HT(truth_usr, pred_usr, topk))
            ctr, correct = metrics.get_ctr(truth_usr, pred_usr, topk)
            correct_list.append(correct)
            ctr_list.append(ctr)
        perf_topk['Recall'] = np.mean(recall_list)
        perf_topk['NDCG'] = np.mean(ndcg_list)
        perf_topk['HitRatio'] = np.mean(ht_list)
        perf_topk['CTR'] = np.mean(ctr_list)
        perf_topk['Correct'] = np.mean(correct_list)
        performance[str(topk)] = perf_topk # string here
    fold_performance[fold] = performance
print(fold_performance)
if not os.path.exists(f'results/{dataset}'):
    os.makedirs(f'results/{dataset}')
results_name = f'results/{dataset}/test.json'
with open(results_name, 'w') as f:
    json.dump(fold_performance, f)
print("Done!!")
# -*- coding: utf-8 -*-
# @Time    : 26/04/2022 09:49
# @Author  : Ming

import pickle
import json
import numpy as np
import argparse
from gensim.models import Word2Vec
import metrics
import random
from tqdm import tqdm
import scipy.stats as stats
import pandas as pd

def get_performance_dict(expl_pred, item_usr_label, item_set):
    # ratio_performance_dict = {'recall':dict(), 'ndcg':dict(), 'hit':dict(), 'ctr':dict(), 'correct':dict()}
    ratio_performance_dict = dict()
    recall_performance = {item: dict() for item in item_set}
    ndcg_performance = {item: dict() for item in item_set}
    hit_performance = {item: dict() for item in item_set}
    ctr_performance = {item: dict() for item in item_set}
    correct_performance = {item: dict() for item in item_set}
    for item in tqdm(item_set):
        rankusr = expl_pred[item]
        truth_usr = item_usr_label[item][0]
        for topk in [50, 100, 200, 500]:
            recall_performance[item][topk] = metrics.get_Recall(truth_usr, rankusr, topk)
            ndcg_performance[item][topk] = metrics.get_NDCG(truth_usr, rankusr, topk)
            hit_performance[item][topk] = metrics.get_HT(truth_usr, rankusr, topk)
            ctr, correct = metrics.get_ctr(truth_usr, rankusr, topk)
            ctr_performance[item][topk] = ctr
            correct_performance[item][topk] = correct
    ratio_performance_dict['recall'] = recall_performance
    ratio_performance_dict['ndcg'] = ndcg_performance
    ratio_performance_dict['hit'] = hit_performance
    ratio_performance_dict['ctr'] = ctr_performance
    ratio_performance_dict['correct'] = correct_performance

    return ratio_performance_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dunnhumby', help='Dataset')
    parser.add_argument('--nextk', type=int, default=5, help='max_basket_num')
    parser.add_argument('--metric', type=str, default='recall', help='max_basket_num')
    parser.add_argument('--topk', type=int, default=50, help='max_basket_num')
    parser.add_argument('--aug_pos', type=int, default=0, help='x')
    parser.add_argument('--adj_neg', type=int, default=0, help='x')
    parser.add_argument('--cfm', type=int, default=0, help='x')
    parser.add_argument('--freq', type=int, default=0, help='x')
    args = parser.parse_args()
    dataset, nextk, metric = args.dataset, args.nextk, args.metric
    candidate_filter = args.cfm
    freq_signal =args.freq
    aug_pos=args.aug_pos
    adjust_neg=args.adj_neg

    with open(f"../data/{dataset}_temp/item_user_label{nextk}.pkl", 'rb') as f:
        item_usr_label = pickle.load(f)
    # load fold info
    with open(f"../data/{dataset}_temp/foldk.pkl", 'rb') as f:
        foldk = pickle.load(f)

    with open(f'../data/{dataset}_temp/basket_seq_filter_all_30.pkl', 'rb') as f:
        usr_seq_b = pickle.load(f)

    with open(f'../data/{dataset}_temp/item_cat_info.pkl', 'rb') as f:
        item_cat_info = pickle.load(f)

    pre_trained_emb_path = f'../data/{dataset}_emb/basket_level_100d.model'
    item_emb_model = Word2Vec.load(pre_trained_emb_path)
    emb_item_set = item_emb_model.wv.index_to_key
    item_set = foldk[0]['val'] + foldk[0]['test']
    item_set = set(item_set)&set(emb_item_set)
    max_uid = np.max(list(item_set))

    expl_pred_file_a = f'test_results_mean/{dataset}/cand1_freq1_aug1_adjust1/expl_rankusr_0.pkl'
    with open(expl_pred_file_a, 'rb') as f:
        expl_pred_a = pickle.load(f)
    if freq_signal == 1:
        expl_pred_file_b = f'test_results_mean/{dataset}/cand1_freq0_aug1_adjust1/expl_rankusr_0.pkl'
        with open(expl_pred_file_b, 'rb') as f:
            expl_pred_b = pickle.load(f)
    if candidate_filter == 1:
        expl_pred_file_b = f'test_results_mean/{dataset}/cand0_freq1_aug1_adjust1/expl_rankusr_0.pkl'
        with open(expl_pred_file_b, 'rb') as f:
            expl_pred_b = pickle.load(f)
    if aug_pos == 1:
        expl_pred_file_b = f'test_results_mean/{dataset}/cand1_freq1_aug0_adjust1/expl_rankusr_0.pkl'
        with open(expl_pred_file_b, 'rb') as f:
            expl_pred_b = pickle.load(f)
    if adjust_neg == 1:
        expl_pred_file_b = f'test_results_mean/{dataset}/cand1_freq1_aug1_adjust0/expl_rankusr_0.pkl'
        with open(expl_pred_file_b, 'rb') as f:
            expl_pred_b = pickle.load(f)

    a_performance_dict = get_performance_dict(expl_pred_a, item_usr_label, item_set)
    b_performance_dict = get_performance_dict(expl_pred_b, item_usr_label, item_set)

    # start significant test
    for topk in [50, 100, 200, 500]:
        print('====================TOPK: {}=================='.format(topk))
        for m in ['recall', 'ndcg', 'hit', 'ctr', 'correct']:
            a_perf_list = []
            b_perf_list = []
            for item in item_set:
                a_perf_list.append(a_performance_dict[m][item][topk])
                b_perf_list.append(b_performance_dict[m][item][topk])
            a_perf_list = np.array(a_perf_list)
            b_perf_list = np.array(b_perf_list)
            performance = pd.DataFrame({'b': b_perf_list,
                                        'a': a_perf_list,
                                        'diff': a_perf_list - b_perf_list})
            # print(performance['diff'])
            print('Metric: {}'.format(m))
            print(stats.ttest_rel(a=a_perf_list, b=b_perf_list))



# for fold in range(5):
#     performance_foldk = []
#     item_test_set = set(foldk[fold][group]) & set(item_set)
#     for item in item_test_set:
#         performance_foldk.append(ratio_performance_dict[metric][threshold][item][k])
#     performance_list.append(np.mean(performance_foldk))


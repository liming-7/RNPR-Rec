# -*- coding: utf-8 -*-
# @Time    : 22/04/2022 14:03
# @Author  : Ming


import pickle
import numpy as np
import argparse
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import metrics
import utils
import random
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dunnhumby', help='Dataset')
    parser.add_argument('--nextk', type=int, default=5, help='max_basket_num')
    parser.add_argument('--metric', type=str, default='recall', help='max_basket_num')
    parser.add_argument('--topk', type=int, default=500, help='max_basket_num')
    args = parser.parse_args()
    dataset, nextk, metric = args.dataset, args.nextk, args.metric

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

    expl_pred_file = f'test_results_mean/{dataset}/cand1_freq1_aug1_adjust1/expl_rankusr_0.pkl'
    with open(expl_pred_file, 'rb') as f:
        expl_pred = pickle.load(f)

    ratio_performance_dict = {'recall':dict(), 'ndcg':dict(), 'hit':dict(), 'ctr':dict(), 'correct':dict()}
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        with open(f'cfm_model/{dataset}/balanced/threshold_{threshold}.pkl', 'rb') as f:
            cand = pickle.load(f)
        recall_performance = {item: dict() for item in item_set}
        ndcg_performance = {item: dict() for item in item_set}
        hit_performance = {item: dict() for item in item_set}
        ctr_performance = {item: dict() for item in item_set}
        correct_performance = {item: dict() for item in item_set}
        for item in tqdm(item_set):
            i_cat = item_cat_info['item_cat_dict'][item]
            expl_usr = expl_pred[item]
            truth_usr = item_usr_label[item][0]
            rankusr = [usr for usr in expl_usr if usr in cand[i_cat]]
            left_expl_usr = list(set(expl_usr)-set(rankusr))
            if len(left_expl_usr) != 0:
                random.shuffle(left_expl_usr)
                rankusr = rankusr + left_expl_usr

            for topk in [50, 100, 200, 500]:
                recall_performance[item][topk] = metrics.get_Recall(truth_usr, rankusr, topk)
                ndcg_performance[item][topk] = metrics.get_NDCG(truth_usr, rankusr, topk)
                hit_performance[item][topk] = metrics.get_HT(truth_usr, rankusr, topk)
                ctr, correct = metrics.get_ctr(truth_usr, rankusr, topk)
                ctr_performance[item][topk] = ctr
                correct_performance[item][topk] = correct
        ratio_performance_dict['recall'][threshold] = recall_performance
        ratio_performance_dict['ndcg'][threshold] = ndcg_performance
        ratio_performance_dict['hit'][threshold] = hit_performance
        ratio_performance_dict['ctr'][threshold] = ctr_performance
        ratio_performance_dict['correct'][threshold] = correct_performance


    for k in [50, 100, 200, 500]:
        print("================TopK: {}===============".format(k))
        for group in ['test', 'val']:
            final_performance_dict = dict()
            for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                performance_list = []
                for fold in range(5):
                    performance_foldk = []
                    item_test_set = set(foldk[fold][group])&set(item_set)
                    for item in item_test_set:
                        performance_foldk.append(ratio_performance_dict[metric][threshold][item][k])
                    performance_list.append(np.mean(performance_foldk))

                final_performance_dict[threshold] = np.mean(performance_list)
            print('--------{}---------'.format(group))
            print(final_performance_dict)


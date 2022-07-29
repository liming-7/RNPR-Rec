# -*- coding: utf-8 -*-
# @Time    : 18/11/2021 18:18
# @Author  : Ming
import sys
import datetime
import json
import pickle
import os

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torchmetrics

import warnings
from tqdm import tqdm
import numpy as np
import utils
import metrics

from rankusr_model_explore_aware import rankusr_model_explore_aware_model
from data_loader import get_dataloader, get_dataset
from utils import convert_to_gpu, convert_all_data_to_gpu, save_model, load_model

import argparse


def get_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dunnhumby', help='Dataset')
    parser.add_argument('--nextk', type=int, default=5, help='max_basket_num')
    parser.add_argument('--avg_type', type=str, default='max', help='pooling max or mean')
    parser.add_argument('--candidate_filter', type=int, default=1, help='0 : no filter, 1: filter')
    parser.add_argument('--aug_pos', type=int, default=1, help='abla: 1 aug; 0 no aug')
    parser.add_argument('--adjust_neg', type=int, default=1, help='abla: 1 adjust; 0 no adjust')
    parser.add_argument('--freq_signal', type=int, default=1, help='abla: 1: use freq; 0: do not use freq')
    parser.add_argument('--epoch', type=int, required=True, help='xx')
    parser.add_argument('--loss_func', type=str, default='bpr', help='bpr or bce loss')

    args = parser.parse_args()
    config['dataset'], config["nextk"], config['avg_type'], config['candidate_filter'], config['avg_type'], config['aug_pos'], config['adjust_neg'] =\
        args.dataset, args.nextk, args.avg_type, args.candidate_filter, args.avg_type, args.aug_pos, args.adjust_neg
    config['freq_signal'], config['loss_func'] = args.freq_signal, args.loss_func
    config['epoch'] = args.epoch

    dataset = config['dataset']
    config['item_category_path'] = f'../data/{dataset}_temp/item_cat_info.pkl'
    config['pretrained_emb_path'] = f'../data/{dataset}_emb/basket_level_100d.model'
    config['train_data_path'] = f'../data/{dataset}_temp/train_filter_30_train.pkl'
    config['origin_data_path'] = f'../data/{dataset}_temp/basket_seq_filter_all_30.pkl'
    config['fold_path'] = f'../data/{dataset}_temp/foldk.pkl'
    with open(config['item_category_path'], 'rb') as f:
        item_cat_info = pickle.load(f)
    config['cat_num'] = item_cat_info['cat_num']
    config['item_num'] = item_cat_info['item_num']

    if torch.cuda.is_available():
        print("CUDA!!")
        device = torch.device('cuda:0')
    else:
        print("CPU!!")
        device = torch.device('cpu')
    config['device'] = device
    sys.stdout.flush()

    config['model_path'] = f"model_{config['avg_type']}/{config['dataset']}/cand{config['candidate_filter']}_freq{config['freq_signal']}_aug{config['aug_pos']}_adjust{config['adjust_neg']}/next{config['nextk']}_model_{config['epoch']}.pkl"
    config['save_folder'] = f"test_results_{config['avg_type']}/{config['dataset']}/cand{config['candidate_filter']}_freq{config['freq_signal']}_aug{config['aug_pos']}_adjust{config['adjust_neg']}/"
    if not os.path.exists(config['save_folder']):
        os.makedirs(config['save_folder'])
    return config

def pred_model(model, pred_data_loader, config):

    warnings.filterwarnings('ignore')
    print(model)
    start_time = datetime.datetime.now()
    model.eval()
    tqdm_pred = tqdm(pred_data_loader, miniters=int(1000))
    usr_id_list = pred_data_loader.dataset.get_usr_id_list()
    item_set = pred_data_loader.dataset.get_item_set()
    item_usr_label, cat_usr_label = pred_data_loader.dataset.get_item_usr_labels()
    max_usr_id = np.max(usr_id_list)
    iu_score_dict = {item: np.zeros(max_usr_id+1) for item in item_set}
    for step, (uid, item, cat, sim_seq, rep_seq, expl_seq) in enumerate(tqdm_pred):
        # print(pos_explore_label, pos_item, pos_cat, pos_emb_seq, pos_incat_emb_seq, pos_incat_pos_seq, pos_incat_freq_seq, pos_incat_exp_freq_seq, pos_freq_len)
        if torch.cuda.is_available():
            uid, item, cat, sim_seq, rep_seq, expl_seq = \
                convert_all_data_to_gpu(uid, item, cat, sim_seq, rep_seq, expl_seq, device=config['device'])
        uid, item, cat, sim_seq, rep_seq, expl_seq = \
            torch.tensor(uid), torch.tensor(item), torch.tensor(cat), torch.tensor(sim_seq,dtype=torch.float32), torch.tensor(rep_seq, dtype=torch.float32), torch.tensor(expl_seq, dtype=torch.float32)

        pred_score = model.prob_prediction(item, cat, sim_seq, rep_seq, expl_seq)
        if torch.cuda.is_available():
            pred_score = pred_score.cpu().detach().numpy()
            item = item.cpu().detach().numpy()
            usr_id = uid.cpu().detach().numpy()
        else:
            pred_score = pred_score.detach().numpy()
            item = item.detach().numpy()
            usr_id = uid.detach().numpy()
        for ind in range(len(pred_score)):
            iu_score_dict[item[ind]][usr_id[ind]] = pred_score[ind]

    get_fold_results(iu_score_dict, repeat_flag=False, item_set=item_set, config=config)
    end_time = datetime.datetime.now()
    print("test cost %d seconds" % (end_time - start_time).seconds)
    ## save pred_results
    # with open(config['item_category_path'], 'rb') as f:
    #     item_cat_info = pickle.load(f)
    # save_rank_usr(iu_score_dict, item_set, item_cat_info, cat_usr_label, item_usr_label, config)
    print("ranked usr saved!")

def get_fold_results(iu_score_dict, repeat_flag, item_set, config):
    # load label
    with open(f"../data/{config['dataset']}_temp/item_user_label{config['nextk']}.pkl", 'rb') as f:
        item_usr_label = pickle.load(f)
    # load fold info
    with open(f"../data/{config['dataset']}_temp/foldk.pkl", 'rb') as f:
        foldk = pickle.load(f)

    with open(config['origin_data_path'], 'rb') as f:
        usr_seq_b = pickle.load(f)

    usr_item_set = dict()
    for usr, bask_seq in usr_seq_b.items():
        usr_item_set[usr] = set([item for bask in bask_seq[:-10] for item in bask])

    print(len(iu_score_dict.keys()))
    # rank top-k users for item
    print('Evaluating .......')
    for group in ['test', 'val']:
        fold_performance = dict()
        for fold in range(5):
            item_test_set = set(foldk[fold][group])&set(item_set)
            performance = dict()
            for topk in [10, 20, 50, 100, 200, 500]:
                perf_topk = dict()
                recall_list = []
                ndcg_list = []
                ht_list = []
                correct_list = []
                ctr_list = []
                for item in item_test_set:
                    # print(item)
                    if repeat_flag:
                        pred_usr = utils.get_pred_user(iu_score_dict[item], topk, t_item=item)  # include repeat
                        truth_usr = item_usr_label[item][0] + item_usr_label[item][1]
                    else:
                        pred_usr = utils.get_pred_user(iu_score_dict[item], topk, t_item=item, repeat_usr=usr_item_set)  # pure explore
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
                performance[str(topk)] = perf_topk  # string here
            fold_performance[fold] = performance
        print(fold_performance)
        results_name = config['save_folder']+f"/{group}_epoch_{config['epoch']}_all.json"
        with open(results_name, 'w') as f:
            json.dump(fold_performance, f)
        print("save to {}".format(results_name))

def save_rank_usr(iu_score_dict, item_set, item_cat_info, cat_usr_label, item_usr_label, config):
    expl_rankusr_dict = dict()
    rankusr_dict = dict()
    for item in tqdm(item_set):
        cat = item_cat_info['item_cat_dict'][item]
        usr_cand = cat_usr_label[cat]['history_usr']
        repeat_usr = item_usr_label[item]['history_usr']
        ranked_usr, ranked_expl_usr = utils.get_rank_usr(iu_score_dict[item], usr_cand, repeat_usr)
        expl_rankusr_dict[item] = ranked_expl_usr
        rankusr_dict[item] = ranked_usr
    expl_rankusr_file = config['save_folder']+f"/expl_rankusr_{config['epoch']}.pkl"
    ranked_usr_file = config['save_folder']+f"/mixed_rankusr_{config['epoch']}.pkl"
    with open(ranked_usr_file, 'wb') as f:
        pickle.dump(rankusr_dict, f)
    with open(expl_rankusr_file, 'wb') as f:
        pickle.dump(expl_rankusr_dict, f)


def pred_performance(config):

    model = rankusr_model_explore_aware_model(config)
    model_path = config['model_path']

    if torch.cuda.is_available():
        model = convert_to_gpu(model, config['device'])
        model = load_model(model, model_path, map_location='gpu')
    else:
        model = load_model(model, model_path, map_location='cpu')
    test_dataset = get_dataset(config, data_type='test')
    pred_data_loader = get_dataloader(test_dataset, config, data_type='test')
    pred_model(model, pred_data_loader, config)

if __name__ == '__main__':
    config = get_config('config.json')
    pred_performance(config)
    print("Finish! And saved.")


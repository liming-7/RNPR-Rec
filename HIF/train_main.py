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
    parser.add_argument('--loss_func', type=str, default='bpr', help='bpr or bce loss')

    args = parser.parse_args()
    config['dataset'], config["nextk"], config['avg_type'], config['candidate_filter'], config['avg_type'], config['aug_pos'], config['adjust_neg'] =\
        args.dataset, args.nextk, args.avg_type, args.candidate_filter, args.avg_type, args.aug_pos, args.adjust_neg
    config['freq_signal'], config['loss_func'] = args.freq_signal, args.loss_func

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

    # device = torch.device('cuda:0')
    # print(device)
    if torch.cuda.is_available():
        print("CUDA!!")
        device = torch.device('cuda:0')
    else:
        print("CPU!!")
        device = torch.device('cpu')
    config['device'] = device
    sys.stdout.flush()

    config['save_folder'] = f"model_{config['avg_type']}/{config['dataset']}/cand{config['candidate_filter']}_freq{config['freq_signal']}_aug{config['aug_pos']}_adjust{config['adjust_neg']}/"
    if not os.path.exists(config['save_folder']):
        os.makedirs(config['save_folder'])
    return config

def train_model(model, dataset, optimizer, config):

    warnings.filterwarnings('ignore')
    print(model)
    print(optimizer)
    device = config['device']
    if torch.cuda.is_available():
        model = convert_to_gpu(model, device)
    start_time = datetime.datetime.now()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    for epoch in range(config['epochs']):

        model.train() # train model
        if epoch > 0:
            dataset.update_train_pairs()
        train_data_loader = get_dataloader(dataset, config, data_type='train')
        tqdm_train = tqdm(train_data_loader, miniters=int(100))
        total_loss = 0.0

        for step, (pos_uid, pos_item, pos_cat, pos_sim_seq, pos_rep_seq, pos_expl_seq,
                   neg_uid, neg_item, neg_cat, neg_sim_seq, neg_rep_seq, neg_expl_seq) in enumerate(tqdm_train):
            if torch.cuda.is_available():
                pos_uid, pos_item, pos_cat, pos_sim_seq, pos_rep_seq, pos_expl_seq, \
                neg_uid, neg_item, neg_cat, neg_sim_seq, neg_rep_seq, neg_expl_seq =\
                    convert_all_data_to_gpu(pos_uid, pos_item, pos_cat, pos_sim_seq, pos_rep_seq, pos_expl_seq, neg_uid, neg_item, neg_cat, neg_sim_seq, neg_rep_seq, neg_expl_seq, device=device)

            pos_uid, pos_item, pos_cat, pos_sim_seq, pos_rep_seq, pos_expl_seq = \
                torch.tensor(pos_uid), torch.tensor(pos_item), torch.tensor(pos_cat), torch.tensor(pos_sim_seq, dtype=torch.float32),\
                torch.tensor(pos_rep_seq, dtype=torch.float32), torch.tensor(pos_expl_seq, dtype=torch.float32)
            neg_uid, neg_item, neg_cat, neg_sim_seq, neg_rep_seq, neg_expl_seq = \
                torch.tensor(neg_uid), torch.tensor(neg_item), torch.tensor(neg_cat), torch.tensor(neg_sim_seq, dtype=torch.float32),\
                torch.tensor(neg_rep_seq, dtype=torch.float32), torch.tensor(neg_expl_seq, dtype=torch.float32)

            optimizer.zero_grad()
            loss = model.calculate_loss(pos_item, pos_cat, pos_sim_seq, pos_rep_seq, pos_expl_seq,
                                        neg_item, neg_cat, neg_sim_seq, neg_rep_seq, neg_expl_seq)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
            optimizer.step()
            total_loss += loss.cpu().data.numpy()
            tqdm_train.set_description(f'epoch: {epoch}, train loss: {total_loss / (step + 1)}', refresh=False)
        scheduler.step(total_loss) # total_loss/(step+1)

        model_path = config['save_folder'] +f"/next{config['nextk']}_model_{epoch}.pkl"
        save_model(model, model_path)
        print(f"best model save as {model_path}!!")
    end_time = datetime.datetime.now()
    print("train cost %d seconds" % (end_time - start_time).seconds)
    return model_path

# def get_predication(config, best_model_path):
#     model = category_pred_model(config)
#     test_data_loader = get_dataloader(config, data_type='test') # future
#
#     model = load_model(model, best_model_path, map_location='cpu')
#     model.eval()


def train(config):

    model = rankusr_model_explore_aware_model(config)
    dataset = get_dataset(config, data_type='train')
    # train_data_loader = get_dataloader(config, data_type='train')
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    train_model(model, dataset, optimizer, config)


if __name__ == '__main__':
    config = get_config('config.json')
    best_model_path = train(config)
    print("Finish! And best model path is {}".format(best_model_path))


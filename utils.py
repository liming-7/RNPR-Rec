# -*- coding: utf-8 -*-
# @Time    : 18/11/2021 18:50
# @Author  : Ming

import json
import torch
import torch.nn.functional as F
import numpy as np
import pickle

# convert data from cpu to gpu, accelerate the running speed
def convert_to_gpu(data, device):
    # if get_attribute('cuda') != -1 and torch.cuda.is_available():
    data = data.to(device)
    return data


def convert_all_data_to_gpu(*data, device=None):
    res = []
    for item in data:
        item = convert_to_gpu(item, device)
        res.append(item)
    return tuple(res)


def convert_train_truth_to_gpu(train_data, truth_data):
    train_data = [[convert_to_gpu(basket) for basket in baskets] for baskets in train_data]
    truth_data = convert_to_gpu(truth_data)
    return train_data, truth_data

# load parameters of model
def load_model(model_object, model_file_path, map_location=None):
    if map_location is None:
        model_object.load_state_dict(torch.load(model_file_path))
    else:
        model_object.load_state_dict(torch.load(model_file_path, map_location=map_location))

    return model_object


# save parameters of the model
def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)


def get_itemset(path, min_num):
    # select items above 50 new users.
    with open(path, 'rb') as f:
        item_usr_dict = pickle.load(f)

    test_train_items = []
    for item, users in item_usr_dict.items():
        if len(item_usr_dict[item][0]) >= min_num:
            test_train_items.append(item)
    # with open('instacart_temp/items_above50-5.pkl', 'wb') as f:
    #     pickle.dump(test_train_items, f)
    return item_usr_dict, test_train_items


def get_pred_user(iu_score, top_k, t_item=None, repeat_usr=None):
    if repeat_usr is None:
        pred_rank_usr = np.argsort(iu_score)[::-1].tolist()[:top_k]
    else:
        num = 0
        pred_rank_usr = []
        pred_rank_usr_all = np.argsort(iu_score)[::-1].tolist()
        for usr in pred_rank_usr_all:
            if usr in repeat_usr.keys():
                if t_item not in repeat_usr[usr]:
                    pred_rank_usr.append(usr)
                    num +=1
                if num>=top_k:
                    break
    return pred_rank_usr

def get_rank_usr(iu_score, usr_cand, repeat_usr):
    ranked_usr_list = np.argsort(iu_score)[::-1].tolist()
    ranked_usr = []
    for usr in ranked_usr_list:
        if usr in usr_cand: # ensure the candidate usr set. repeat category.
            ranked_usr.append(usr)
        if iu_score[usr] == 0:
            break

    ranked_usr_remove_repeat = []
    for usr in ranked_usr:
        if usr not in repeat_usr:
            ranked_usr_remove_repeat.append(usr)
    return ranked_usr, ranked_usr_remove_repeat

# read product category info，get item's cat, get cat's items

# def get_item_cat(item_id, item_cat_info, item_col_name, cat_col_name, reindex_dict=None):
#     return item_cat_info[item_cat_info[item_col_name].isin([item_id])][cat_col_name].iat[0]

def get_item_cat(item, cat_items_dict):
    for cat in cat_items_dict.keys():
        if item in cat_items_dict[cat]:
            return cat
    raise KeyError

def get_cat_items(cat_id, item_cat_info, item_col_name, cat_col_name, reindex_dict=None):
    if reindex_dict is not None:
        items = item_cat_info[item_cat_info[cat_col_name].isin([cat_id])][item_col_name].unique()
        if items is None:
            return []
        else:
            return [reindex_dict[item] for item in items if item in reindex_dict.keys()]
    else:
        return item_cat_info[item_cat_info[cat_col_name].isin([cat_id])][item_col_name].unique()


def get_all_cats(item_cat_info, cat_col_name):
    return item_cat_info[cat_col_name].unique()

def consine_sim(vec1, vec2):
    if np.sum(vec1) == 0 or np.sum(vec2) == 0:
        return 0
    else:
        return np.dot(vec1, vec2)/(np.sqrt(np.sum(vec1**2))*np.sqrt(np.sum(vec2**2)))
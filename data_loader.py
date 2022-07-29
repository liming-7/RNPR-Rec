# -*- coding: utf-8 -*-
# @Time    : 15/11/2021 20:47
# @Author  : Ming
# @Author  : Ming
import pickle
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
import itertools
import numpy as np
import random
from sklearn.preprocessing import normalize
from gensim.models import Word2Vec
import math
from utils import convert_to_gpu, convert_all_data_to_gpu, save_model, load_model
from candidate_filter_model import candidate_filter_model
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from tqdm import tqdm


class BasketFreqDataset(Dataset):

    def __init__(self, config, type=None):

        self.batch_size = config['batch_size']
        self.emb_max_len = config['emb_max_len']
        self.freq_max_len = config['freq_max_len']
        self.nextk = config['nextk']
        self.aug_pos = config['aug_pos']
        self.adjust_neg = config['adjust_neg']
        self.train_data_path = config['train_data_path']
        self.origin_data_path = config['origin_data_path']
        self.item_category_path = config['item_category_path']
        self.fold_path = config['fold_path']
        self.avg_type = config['avg_type']
        self.candidate_filter = config['candidate_filter']
        self.emb_dim = config['embed_dim']
        self.type = type

        # load train data and test data.
        if self.type == 'train':
            with open(self.train_data_path, 'rb') as f:
                data = pickle.load(f)
            self.data_dict = data
        elif self.type == 'test' or self.type == 'val':
            with open(self.origin_data_path, 'rb') as f:
                data = pickle.load(f)
            self.data_dict = dict()
            for uid, bask_seq in data.items():
                self.data_dict[uid] = bask_seq[:-10+self.nextk] # did not set the mex length

        with open(self.item_category_path, 'rb') as f:
            self.item_cat_info = pickle.load(f)
        with open(self.fold_path, 'rb') as f:
            fold_usr = pickle.load(f)

        self.item_set = set(fold_usr[0]['val'] + fold_usr[0]['test']) # get item set
        self.cat_set = set(self.item_cat_info['cat_items_dict'].keys())
        self.usr_set = set(self.data_dict.keys())

        # load pre-trained embedding
        self.pretrained_item_emb = Word2Vec.load(config['pretrained_emb_path'])
        self.emb_item_list = self.pretrained_item_emb.wv.index_to_key

        # compute the initial information
        self.item_usr_label, self.cat_usr_label = self.get_item_cat_usr_label(self.data_dict)

        # pre-compute the frequency info.
        self.usr_freq_seq_dict = dict() # two dim array: max_len * cat_num, pre-computed user cat frequency info.
        self.usr_exp_freq_seq_dict = dict()
        self.usr_bask_emb_seq = dict()
        self.usr_overall_emb_seq = dict()
        self.item_exp_aug_usr = {item: dict() for item in self.item_set} # indicates the first purchase of the given item.
        for usr, bask_seq in tqdm(self.data_dict.items()):
            h_bask_seq, f_bask_seq = bask_seq[:-self.nextk], bask_seq[-self.nextk:]
            cat_freq_seq = []
            exp_cat_freq_seq = []
            overall_emb_seq = []
            bask_emb_seq = []
            h_item_set = set() # history item of the user, will update per basket
            h_item_list = []
            exp_ind = 0
            for bask in h_bask_seq: # from the first to the last
                cat_freq_vector = np.zeros(self.item_cat_info['cat_num'])
                exp_cat_freq_vector = np.zeros(self.item_cat_info['cat_num'])
                # aug positive instance for exp
                for item in bask:
                    i_cat = self.item_cat_info['item_cat_dict'][item]
                    cat_freq_vector[i_cat] += 1
                    if item not in h_item_set: # indicates this is a new item for this basket
                        exp_cat_freq_vector[i_cat] += 1
                        if item in self.item_set:
                            self.item_exp_aug_usr[item][usr] = exp_ind
                # category freq sequence
                for cat_ind in range(self.item_cat_info['cat_num']):
                    # default sqrt
                    if cat_freq_vector[cat_ind] > 1:
                        cat_freq_vector[cat_ind] = math.sqrt(cat_freq_vector[cat_ind])

                cat_freq_seq.append(cat_freq_vector)
                exp_cat_freq_seq.append(exp_cat_freq_vector)
                exp_ind += 1
                h_item_set.update(set(bask))
                h_item_list = h_item_list+bask

                if len(set(bask)&set(self.emb_item_list)) !=0:
                    bask_emb_seq.append(self.get_pool_embedding(bask))
                else:
                    bask_emb_seq.append(0)

                if len(set(h_item_list)&set(self.emb_item_list)) != 0:
                    overall_emb_seq.append(self.get_pool_embedding(h_item_list))
                else:
                    overall_emb_seq.append(0)

            self.usr_freq_seq_dict[usr] = np.array(cat_freq_seq)
            self.usr_exp_freq_seq_dict[usr] = np.array(exp_cat_freq_seq)
            self.usr_bask_emb_seq[usr] = bask_emb_seq
            self.usr_overall_emb_seq[usr] = overall_emb_seq

        if self.type == 'train':
            # pre-select the positive and negative instances for each item
            self.item_pos_feature_dict = dict()
            self.item_neg_feature_dict = dict()
            self.item_pos_dict = dict()
            self.item_neg_dict = dict()
            for item in tqdm(self.item_set & set(self.emb_item_list)):
                i_cat = self.item_cat_info['item_cat_dict'][item]
                if self.candidate_filter == 1:
                    candidate_set = set(self.cat_usr_label[i_cat]['history_usr'])  # could change the candidate set
                else:
                    candidate_set = random.choices(list(self.usr_set), k=len(self.cat_usr_label[i_cat]['history_usr']))
                    candidate_set = set(candidate_set)
                if self.aug_pos == 1:
                    aug_usr_set = set([usr for usr in self.item_exp_aug_usr[item].keys() if self.item_exp_aug_usr[item][usr] >= 10]) # get aug_usr_set
                    postive_usr_set = (set(self.item_usr_label[item]['future_usr']) - set(self.item_usr_label[item]['history_usr'])) | aug_usr_set
                    postive_usr_list = self.get_usr_set_nan(item, postive_usr_set, aug_usr_set, positive=True)
                else:
                    postive_usr_set = set(self.item_usr_label[item]['future_usr']) - set(self.item_usr_label[item]['history_usr'])
                    postive_usr_list = self.get_usr_set_nan(item, postive_usr_set, positive=True)
                if len(postive_usr_list) == 0:
                    continue
                self.item_pos_feature_dict[item] = self.get_usr_features(postive_usr_list, item, i_cat)
                self.item_pos_dict[item] = postive_usr_list

                negtive_usr_set = candidate_set - set(self.item_usr_label[item]['history_usr']) - postive_usr_set
                if self.adjust_neg == 1:
                    negtive_usr_set = self.filter_neg_set_by_length(negtive_usr_set)  # control
                negtive_usr_list = self.get_usr_set_nan(item, negtive_usr_set, positive=False) # if we want to accerlate the training, maybe do a sampling on this negative set.
                if len(negtive_usr_list) == 0:
                    continue
                self.item_neg_feature_dict[item] = self.get_usr_features(negtive_usr_list, item, i_cat)
                self.item_neg_dict[item] = negtive_usr_list

            self.train_uid_pos, self.train_uid_neg, self.train_item = self.get_train_pairs()

        elif self.type == 'test' or self.type == 'val':
            self.test_item_feature_dict = dict()
            self.item_usr_cand_dict = dict()
            for item in tqdm(self.item_set & set(self.emb_item_list)):
                i_cat = self.item_cat_info['item_cat_dict'][item]
                ## you need you change this part?
                candidate_set = set(self.cat_usr_label[i_cat]['history_usr'])# candidate
                # candidate_set = set(self.usr_set)# candidate
                item_candidate_set = candidate_set - set(self.item_usr_label[item]['history_usr'])
                item_candidate_set = self.get_usr_set_nan(item, item_candidate_set, positive=True)
                if len(item_candidate_set) == 0:
                    continue
                self.test_item_feature_dict[item] = self.get_usr_features(item_candidate_set, item, i_cat)
                self.item_usr_cand_dict[item] = item_candidate_set
            self.test_uid, self.test_item = [], []
            for item in tqdm(self.item_set & set(self.emb_item_list)):
                self.test_item = self.test_item + [item for _ in range(len(self.item_usr_cand_dict[item]))]
                self.test_uid = self.test_uid + list(self.item_usr_cand_dict[item])

    def __getitem__(self, index):
        '''
        :param index:
        :return: category info, freq each category
        '''

        if self.type == 'train':
            item = self.train_item[index]
            i_cat = self.item_cat_info['item_cat_dict'][item]
            pos_uid = self.train_uid_pos[index]
            pos_sim_seq, pos_rep_seq, pos_expl_seq = self.item_pos_feature_dict[item]['sim_seq'][pos_uid], self.item_pos_feature_dict[item]['rep_seq'][pos_uid], self.item_pos_feature_dict[item]['expl_seq'][pos_uid]

            neg_uid = self.train_uid_neg[index]
            neg_sim_seq, neg_rep_seq, neg_expl_seq = self.item_neg_feature_dict[item]['sim_seq'][neg_uid], self.item_neg_feature_dict[item]['rep_seq'][neg_uid], self.item_neg_feature_dict[item]['expl_seq'][neg_uid]
            return pos_uid, item, i_cat, pos_sim_seq, pos_rep_seq, pos_expl_seq,\
                   neg_uid, item, i_cat, neg_sim_seq, neg_rep_seq, neg_expl_seq

        elif self.type == 'test' or self.type == 'val':
            uid = self.test_uid[index]
            item = self.test_item[index]
            i_cat = self.item_cat_info['item_cat_dict'][item]
            sim_seq, rep_seq, expl_seq = self.test_item_feature_dict[item]['sim_seq'][uid], self.test_item_feature_dict[item]['rep_seq'][uid], self.test_item_feature_dict[item]['expl_seq'][uid]
            return uid, item, i_cat, sim_seq, rep_seq, expl_seq

    def __len__(self):
        if self.type == 'train':
            return len(self.train_uid_pos)
        elif self.type == 'test' or self.type == 'val':
            return len(self.test_uid)

    def get_usr_id_list(self): # all users, not only the candidates
        return list(self.data_dict.keys())

    def get_item_cat_usr_label(self, data_dict):

        item_usr_label = {item: {'history_usr': [], 'future_usr': []} for item in self.item_set}
        cat_usr_label = {cat: {'history_usr': [], 'future_usr': [], 'future_exp_usr': [], 'future_rep_usr': []} for cat in self.cat_set}
        for uid, bask_seq in data_dict.items():
            h_items, f_items = set([item for bask in bask_seq[:-self.nextk] for item in bask]),\
                               set([item for bask in bask_seq[-self.nextk:] for item in bask])
            h_cats, f_cats = set([self.item_cat_info['item_cat_dict'][item] for item in h_items]),\
                             set([self.item_cat_info['item_cat_dict'][item] for item in f_items])
            f_rep_items, f_exp_items = f_items&h_items, f_items-h_items
            f_rep_cats, f_exp_cats = set([self.item_cat_info['item_cat_dict'][item] for item in f_rep_items]),\
                                   set([self.item_cat_info['item_cat_dict'][item] for item in f_exp_items])
            for item in h_items&self.item_set:
                item_usr_label[item]['history_usr'].append(uid)
            for item in f_items&self.item_set:
                item_usr_label[item]['future_usr'].append(uid)
            for cat in h_cats:
                cat_usr_label[cat]['history_usr'].append(uid)
            for cat in f_cats:
                cat_usr_label[cat]['future_usr'].append(uid)
            for cat in f_rep_cats:
                cat_usr_label[cat]['future_rep_usr'].append(uid)
            for cat in f_exp_cats:
                cat_usr_label[cat]['future_exp_usr'].append(uid)
        return item_usr_label, cat_usr_label

    def get_item_set(self):
        return self.item_set & set(self.emb_item_list)

    def get_cat_set(self):
        return self.cat_set

    def get_item_usr_labels(self):
        return self.item_usr_label, self.cat_usr_label

    def get_usr_set(self):
        return self.usr_set

    def get_item_cat_info(self):
        return self.item_cat_info

    def get_usr_features(self, usr_list, item, i_cat):

        sim_seq_dict = dict()
        rep_seq_dict = dict()
        expl_seq_dict = dict()
        for usr_id in usr_list:
            user_data = self.data_dict[usr_id]
            # construct label, get h_data and freq seq: since we user tricks to aug positive explore users
            if self.type == 'train':
                if usr_id in self.item_usr_label[item]['history_usr']: # augument user
                    pos_exp_ind = self.item_exp_aug_usr[item][usr_id] # get the position of the first appreance of the item.
                    incat_freq_seq = self.usr_freq_seq_dict[usr_id][:pos_exp_ind, i_cat]
                    incat_exp_freq_seq = self.usr_exp_freq_seq_dict[usr_id][:pos_exp_ind, i_cat]
                    bask_emb_seq = self.usr_bask_emb_seq[usr_id][:pos_exp_ind]
                    overall_emb = self.usr_overall_emb_seq[usr_id][pos_exp_ind-1]
                elif usr_id in self.item_usr_label[item]['future_usr']: # original positive user
                    incat_freq_seq = self.usr_freq_seq_dict[usr_id][:, i_cat]
                    incat_exp_freq_seq = self.usr_exp_freq_seq_dict[usr_id][:, i_cat]
                    bask_emb_seq = self.usr_bask_emb_seq[usr_id]
                    overall_emb = self.usr_overall_emb_seq[usr_id][-1]
                else:
                    # negative users
                    if self.adjust_neg == 1:
                        incat_freq_seq = self.usr_freq_seq_dict[usr_id][:, i_cat][:-self.freq_max_len+self.nextk] #-15 not -20, + self.nextk since this pre-compute sequence is historical data
                        incat_exp_freq_seq = self.usr_exp_freq_seq_dict[usr_id][:, i_cat][:-self.freq_max_len+self.nextk]
                        bask_emb_seq = self.usr_bask_emb_seq[usr_id][:-self.freq_max_len+self.nextk]
                        overall_emb = self.usr_overall_emb_seq[usr_id][-self.freq_max_len+self.nextk-1]
                    else:
                        incat_freq_seq = self.usr_freq_seq_dict[usr_id][:, i_cat]
                        incat_exp_freq_seq = self.usr_exp_freq_seq_dict[usr_id][:, i_cat]
                        bask_emb_seq = self.usr_bask_emb_seq[usr_id]
                        overall_emb = self.usr_overall_emb_seq[usr_id][-1]
            elif self.type == 'test' or self.type == 'val':
                incat_freq_seq = self.usr_freq_seq_dict[usr_id][:, i_cat]
                incat_exp_freq_seq = self.usr_exp_freq_seq_dict[usr_id][:, i_cat]
                bask_emb_seq = self.usr_bask_emb_seq[usr_id]
                overall_emb = self.usr_overall_emb_seq[usr_id][-1]

            # compute frequency feature
            if len(incat_freq_seq) >= self.freq_max_len:
                incat_freq_seq = incat_freq_seq[::-1][:self.freq_max_len].copy()
                incat_exp_freq_seq = incat_exp_freq_seq[::-1][:self.freq_max_len].copy()
            else:
                pad_num = self.freq_max_len - len(incat_freq_seq)
                incat_freq_seq = np.concatenate((incat_freq_seq[::-1], np.zeros(pad_num)), axis=0).copy()
                incat_exp_freq_seq = np.concatenate((incat_exp_freq_seq[::-1], np.zeros(pad_num)), axis=0).copy()

            # pred_prob = self.model.prob_prediction(torch.tensor([i_cat]), torch.tensor([incat_freq_seq], dtype=torch.float32))
            # pred_prob = pred_prob.detach().item()
            #
            # if np.sum(incat_exp_freq_seq[:self.nextk*2])>0: # set a window!!!
            #     expl_indicator = 1
            # else:
            #     expl_indicator = 0

            # compute sim sequence
            item_emb = self.pretrained_item_emb.wv[item]
            sim_seq = []
            overall_sim = 1 - distance.cosine(item_emb, overall_emb)
            for bask_emb in bask_emb_seq:
                if type(bask_emb) is np.ndarray:
                    sim_seq.append(1 - distance.cosine(item_emb, bask_emb))
                else:
                    sim_seq.append(overall_sim)
            if len(sim_seq) >= self.freq_max_len:
                sim_seq = sim_seq[::-1][:self.freq_max_len]
            else:
                pad_num = self.freq_max_len - len(sim_seq)
                sim_seq = np.concatenate((sim_seq[::-1], np.asarray([overall_sim for _ in range(pad_num)])), axis=0)
            sim_seq_dict[usr_id] = np.array(sim_seq)
            rep_seq_dict[usr_id] = np.array(incat_freq_seq)
            expl_seq_dict[usr_id] = np.array(incat_exp_freq_seq)
        feat_dict = {'sim_seq':sim_seq_dict, 'rep_seq':rep_seq_dict, 'expl_seq':expl_seq_dict}
        return feat_dict

    def get_pool_embedding(self, item_list):
        emb_list = []
        for item in item_list:
            if item in self.emb_item_list:
                emb_list.append(self.pretrained_item_emb.wv[item])
        if len(emb_list) == 0:
            raise ValueError
        # return np.average(emb_list, axis=0) #average
        if self.avg_type == 'mean':
            return np.average(emb_list, axis=0) #average
        elif self.avg_type == 'max':
            return np.max(emb_list, axis=0)

    def get_usr_set_nan(self, item, usrset, augusrset=set(), positive=True):
        # to ensure the user's item sequence have pre-trained item embedding.
        usrset_list = []
        for usr in usrset|augusrset: # this is a mistake made by previous version
            user_data = self.data_dict[usr]
            if positive:
                if usr in augusrset:
                    h_usr_seq = user_data[:self.item_exp_aug_usr[item][usr]]
                else:
                    h_usr_seq = user_data[:-self.nextk]
            elif self.adjust_neg == 1:
                h_usr_seq = user_data[:-self.emb_max_len]
            else:
                h_usr_seq = user_data[:-self.nextk]
            h_item_set = set([item for bask in h_usr_seq for item in bask])
            if len(h_item_set & set(self.emb_item_list)) > 0:
                usrset_list.append(usr)
        return usrset_list

    def get_train_pairs(self):
        pos_uid = []
        neg_uid = []
        train_item = []
        for item in self.item_set & set(self.emb_item_list):
            if item not in self.item_pos_dict.keys() or item not in self.item_neg_dict.keys():
                continue
            sampled_postive_usr_set = random.choices(self.item_pos_dict[item], k=self.batch_size)
            sampled_negative_usr_set = random.choices(self.item_neg_dict[item], k=self.batch_size)
            pos_uid = pos_uid + sampled_postive_usr_set
            neg_uid = neg_uid + sampled_negative_usr_set
            item_list = [item for _ in range(self.batch_size)]
            train_item = train_item + item_list
        return pos_uid, neg_uid, train_item

    def update_train_pairs(self):
        self.train_uid_pos, self.train_uid_neg, self.train_item = self.get_train_pairs()

    def filter_neg_set_by_length(self, neg_set):
        train_neg = set()
        for usr in neg_set:
            data = self.data_dict[usr]
            if len(data) >= 40:
                train_neg.add(usr)
        return train_neg

    def get_test_item_cand(self):
        uid = []
        test_item = []
        item_cand_dict = dict()
        for item in self.item_set & set(self.emb_item_list):
            i_cat = self.item_cat_info['item_cat_dict'][item]
            i_candidate_set = set(self.cat_usr_label[i_cat]['history_usr']) - set(self.item_usr_label[item]['history_usr'])
            i_usr_list = self.get_usr_set_nan(item, i_candidate_set)
            item_cand_dict[item] = i_usr_list
            uid = uid + i_usr_list
            item_list = [item for _ in range(len(i_usr_list))]
            test_item = test_item + item_list
        return uid, test_item, item_cand_dict

    def cosine_sim(self, vec1, vec2):
        if np.sum(vec1) == 0 or np.sum(vec2) == 0:
            return 0
        else:
            return np.dot(vec1, vec2) / (np.sqrt(np.sum(vec1 ** 2)) * np.sqrt(np.sum(vec2 ** 2)))



def collate_set_across_user(batch_data):
    # gather users
    ret = list()
    for idx, uid_data in enumerate(zip(*batch_data)):
        ret.append(torch.tensor(uid_data))
    return tuple(ret)

def get_dataset(config, data_type):
    dataset = BasketFreqDataset(config, data_type)
    return dataset

def get_dataloader(dataset, config, data_type):
    # dataset = BasketFreqDataset(config, data_type)
    print(f'{data_type} data length -> {len(dataset)}')
    print('batch size: {}'.format(config['batch_size']))
    if data_type == 'train':
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=config['batch_size'],
                                 shuffle=False, #
                                 drop_last=False)
    if data_type == 'test' or data_type == 'val':
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=256,
                                 shuffle=False, #
                                 drop_last=False)
    print('Get the dataloader!')
    return data_loader
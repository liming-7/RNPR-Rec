import numpy as np
import math
#note ground truth is vector, rank_list is the sorted item index.

def label2vec(label_list, input_size):
    #label_list -> list
    #input_size -> item number
    label_vec = np.zeros(input_size)
    for label in label_list:
        label_vec[label]=1
    return label_vec

def vec2label(label_vec):

    # label_vec -> vector
    label_list = []
    for i in range(len(label_vec)):
        if label_vec[i] == 1:
            label_list.append(i)
    return label_list

def get_repeat_explore(repeat_list, pred_rank_list, k):
    count = 0
    repeat_cnt = 0.0
    for pred in pred_rank_list:
        if count >= k:
            break
        if pred in repeat_list:
            repeat_cnt += 1
        count += 1
    repeat_ratio = repeat_cnt/k
    return repeat_ratio, 1-repeat_ratio

# truth_list -> list
# pred_rank_list -> list (prediction of the model)
def get_DCG(truth_list, pred_rank_list, k):
    count = 0
    dcg = 0.0
    for pred in pred_rank_list:
        if count >= k:
            break
        if pred in truth_list:
            dcg += (1.0)/math.log2(count+1+1)
        count += 1
    return dcg

def get_NDCG(truth_list, pred_rank_list, k):
    dcg = get_DCG(truth_list, pred_rank_list, k)
    truth_num = len(truth_list)
    if truth_num > k:
        idcg = get_DCG(truth_list, truth_list, k)
    else:
        idcg = get_DCG(truth_list, truth_list, truth_num)
    ndcg = dcg / idcg
    return ndcg

def get_HT(truth_list, pred_rank_list, k):
    count = 0
    for pred in pred_rank_list:
        if count >= k:
            break
        if pred in truth_list:
            return 1.0
        count += 1
    return 0.0

def get_Recall(truth_list, pred_rank_list, k):
    truth_num = len(truth_list)
    count = 0
    correct = 0.0
    for pred in pred_rank_list:
        if count >= k:
            break
        if pred in truth_list:
            correct += 1
        count += 1
    recall = correct/truth_num
    return recall

def get_ctr(truth_list, pred_rank_list, k):
    truth_num = len(truth_list)
    count = 0
    correct = 0.0
    for pred in pred_rank_list:
        if count >= k:
            break
        if pred in truth_list:
            correct += 1
        count += 1
    pred_cnt = min(len(pred_rank_list), k)
    # pred_cnt = k
    ctr = correct / pred_cnt
    return ctr, correct


def get_precision_recall_F1(truth_list, pred_rank_list):

    truth = len(truth_list)
    positive = len(pred_rank_list)
    correct = len(set(truth_list)&set(pred_rank_list))

    precision = float(correct)/positive
    recall = float(correct)/truth

    if correct == 0:
        f1 = 0
    else:
        f1 = 2*precision*recall/(precision+recall)
    return precision, recall, f1, correct

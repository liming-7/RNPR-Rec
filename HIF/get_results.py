# -*- coding: utf-8 -*-
# @Time    : 07/11/2021 19:37
# @Author  : Ming

import pickle
import json
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dunnhumby', help='Dataset')
    parser.add_argument('--avg_type', type=str, default='mean', help='x')
    parser.add_argument('--aug_pos', type=int, default=1, help='x')
    parser.add_argument('--adj_neg', type=int, default=1, help='x')
    parser.add_argument('--cfm', type=int, default=1, help='x')
    parser.add_argument('--freq', type=int, default=1, help='x')
    args = parser.parse_args()
    dataset = args.dataset
    candidate_filter = args.cfm
    freq_signal =args.freq
    aug_pos=args.aug_pos
    adjust_neg=args.adj_neg
    val_can_num = 50
    avg_type = args.avg_type

    def get_performance_dict(results_data):
        all_gather_dict = dict()
        # for can_num in [20, 50, 100, 200, 500]:
        for can_num in [20, 50, 100, 200, 500]:
            recall = []
            ndcg = []
            ihr = []
            ihn = []
            gather_dict = dict()
            for foldk in range(5):
                result = results_data[str(foldk)][str(can_num)]
                recall.append(result['Recall'])
                ndcg.append(result['NDCG'])
                ihr.append(result['HitRatio'])
                ihn.append(result['Correct'])
            gather_dict['Recall'] = np.mean(recall)
            gather_dict['NDCG'] = np.mean(ndcg)
            gather_dict['IHR'] = np.mean(ihr)
            gather_dict['IHN'] = np.mean(ihn)
            all_gather_dict[can_num] = gather_dict
        return all_gather_dict


    print(dataset)
    save_folder = f"test_results_{avg_type}/{dataset}/cand{candidate_filter}_freq{freq_signal}_aug{aug_pos}_adjust{adjust_neg}/"

    val_best_performance = 0
    for epoch in range(7):
        val_results_file = f'test_results_{avg_type}/{dataset}/cand{candidate_filter}_freq{freq_signal}_aug{aug_pos}_adjust{adjust_neg}/val_epoch_{epoch}.json'
        with open(val_results_file, 'r') as f:
            val_result_data = json.load(f)
        val_performance_dict = get_performance_dict(val_result_data)

        test_results_file = f'test_results_{avg_type}/{dataset}/cand{candidate_filter}_freq{freq_signal}_aug{aug_pos}_adjust{adjust_neg}/test_epoch_{epoch}.json'
        with open(test_results_file, 'r') as f:
            test_result_data = json.load(f)
        test_performance_dict = get_performance_dict(test_result_data)
        if val_performance_dict[val_can_num]['Recall'] > val_best_performance:
            val_best_performance = val_performance_dict[val_can_num]['Recall']
            print('Epoch:{}'.format(epoch))
            print(test_performance_dict)



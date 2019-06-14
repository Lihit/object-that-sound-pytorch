import os
import numpy as np
from sklearn.metrics import jaccard_similarity_score
import pandas as pd
from skimage.transform import resize
from sklearn.model_selection import BaseCrossValidator, train_test_split
from PIL import Image
import cv2
import sys
from datetime import datetime
import torch
import shutil
from collections import defaultdict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  # stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None:
            mode = 'w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message:
            is_file = 0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            # time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return '%2d hr %02d min' % (hr, min)

    elif mode == 'sec':
        t = int(t)
        min = t // 60
        sec = t % 60
        return '%2d min %02d sec' % (min, sec)

    else:
        raise NotImplementedError


def dcg_score(anchor_classes, retrieve_classes_list, mat_rel, genre_rel):
    y_true = []
    for item_list in retrieve_classes_list:
        tmp = []
        for anchor_classes_i in anchor_classes:
            i = genre_rel.index(anchor_classes_i)
            for item_list_j in item_list:
                j = genre_rel.index(item_list_j)
                tmp.append(mat_rel[i, j])
        y_true.append(max(tmp))
    y_true = np.array(y_true)
    y_true_sort = np.sort(y_true)[::-1]
    s1 = np.sum((2 ** y_true - 1) / (np.log2(np.arange(len(y_true)) + 2)))
    s2 = np.sum((2 ** y_true_sort - 1) / (np.log2(np.arange(len(y_true_sort)) + 2)))  # norm
    return s1 / s2


def nDCG_k(anchor_feats, positive_feats, anchor_classes, k, mat_rel, genre_rel):
    nDCG_k_ret = dict()
    for i in range(anchor_feats.shape[0]):
        feats_sub = np.sum(np.square(anchor_feats[i, :] - positive_feats), axis=1)
        topk_list = np.argsort(feats_sub)[:k]
        nDCG_k_ret[tuple(anchor_classes[i])] = [anchor_classes[j] for j in topk_list]
    nDCG_k_score = [dcg_score(key, nDCG_k_ret[key], mat_rel, genre_rel) for key in nDCG_k_ret]
    return np.mean(nDCG_k_score)


def top_k(anchor_feats, positive_feats, k_list):
    topk_acc_ret = defaultdict(list)
    for i in range(anchor_feats.shape[0]):
        feats_sub = np.sum(np.square(anchor_feats[i, :] - positive_feats), axis=1)
        for k in k_list:
            topk_list = np.argsort(feats_sub)[:k]
            topk_acc = 1 if i in topk_list else 0
            topk_acc_ret['top%d_acc' % k].append(topk_acc)
    topk_acc_ret = {key: np.mean(topk_acc_ret[key]) for key in topk_acc_ret}
    return topk_acc_ret


def top_k_percent(test_df, anchor_feats, positive_feats, k_list):
    topk_acc_ret = defaultdict(list)
    for i in range(anchor_feats.shape[0]):
        feats_sub = np.sum(np.square(anchor_feats[i, :] - positive_feats), axis=1)
        for k in k_list:
            k_num = int(k / 100.0 * positive_feats.shape[0])
            topk_list = np.argsort(feats_sub)[:k_num]
            topk_acc = 1 if i in topk_list else 0
            topk_acc_ret['top%d_acc' % k].append(topk_acc)
    topk_acc_ret = {key: np.mean(topk_acc_ret[key]) for key in topk_acc_ret}
    return topk_acc_ret


def StringEditDistance(word1, word2):
    """
    :type word1: str
    :type word2: str
    :rtype: int
    """
    m = len(word1)
    n = len(word2)
    dp = [[0 for __ in range(m + 1)] for __ in range(n + 1)]
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(n + 1):
        dp[i][0] = i
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            onemore = 1 if word1[j - 1] != word2[i - 1] else 0
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + onemore)
    return dp[n][m]


if __name__ == '__main__':
    word1 = "acoustic guitar"
    word2 = "drums"
    ret = StringEditDistance(word1, word2)
    print(ret)

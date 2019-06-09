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


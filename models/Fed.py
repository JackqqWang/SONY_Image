#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedFreq(w1, w2, w3): #5,3,2
    w_avg = copy.deepcopy(w1)
    # print(w_avg.keys())
    for k in w_avg.keys():
        for i in range(4): #because w_avg count 1
            w_avg[k] += w1[k]
        for i in range(3):
            w_avg[k] += w2[k]
        for i in range(2):
            w_avg[k] += w3[k]
        w_avg[k] = torch.div(w_avg[k], 10)
    return w_avg
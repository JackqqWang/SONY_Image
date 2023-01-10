#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=int, default=1, help="for test")
    parser.add_argument('--repeat', type=int, default=3, help="repeat experiment")
    # data
    parser.add_argument('--label_rate', type=float, default=0.1, help="the fraction of labeled data")
    # parser.add_argument('--frac_train_test', type=float, default=0.8, help="the fraction of trainset and testset")
    

    # FL setting
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--epochs', type=int, default=200, help="rounds of training")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--init_epochs', type=int, default=100, help="the initial rounds of training, no client-side training")
    parser.add_argument('--iid', type=str, default ='iid', help='whether i.i.d or not')

    # training
    parser.add_argument('--local_bs', type=int, default=256, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")

    # pseduo label
    parser.add_argument('--threshold', type=float, default=0.7, help="the threshold of generating pseduo labels")

    args = parser.parse_args()
    return args

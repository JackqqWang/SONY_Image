#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
import pandas as pd 
from scipy import stats

# from sklearn.preprocessing import LabelEncoder
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
# import seaborn as sns
from matplotlib.gridspec import GridSpec
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from time import time
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import copy
import numpy as np
import torch
import pandas as pd
from data.sampling import getDataset, iid, noniid
from utils.options import args_parser
from models.Update import ClientUpdate_fedmatch, ServerUpdate_fedavg, batch_generator
from models.Nets import QuickModel, CNNCifar, CNNMnist, LSTM
from models.Fed import FedAvg
from models.test import test
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

from data.randaugment import RandomTranslateWithReflect

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, pseudo_label = None):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.pseudo_label = pseudo_label

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if self.pseudo_label != None:
            label = int(self.pseudo_label[self.idxs[item]]) 
        return image, label

if __name__ == '__main__':
    args = args_parser()

    avg_time_list, avg_loss_server_list, avg_loss_train, avg_acc_test, avg_f1_test, avg_auc_test = [0] * args.epochs, [0] * args.epochs, [0] * args.epochs, [0] * args.epochs, [0] * args.epochs, [0] * args.epochs
    for _ in range(args.repeat):
        args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

        trans_cifar_strong = transforms.Compose([
            RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        trans_cifar_weak = transforms.Compose([
            RandomTranslateWithReflect(1),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        trans_cifar_test = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        dataset_train = datasets.CIFAR10('../dataset/cifar', train=True, download=True, transform=trans_cifar_strong)
        dataset_train_weak = datasets.CIFAR10('../dataset/cifar', train=True, download=True, transform=trans_cifar_weak)
        dataset_test = datasets.CIFAR10('../dataset/cifar', train=False, download=True, transform=trans_cifar_test)

        if args.iid == 'iid':
            dict_users_labeled, dict_users_unlabeled, _ = iid(dataset_train, args.num_users, args.label_rate)
        else:
            dict_users_labeled, dict_users_unlabeled, _ = noniid(dataset_train, args.num_users, args.label_rate)

        print('labeled data:', len(dict_users_labeled))
        print('unlabeled data:', len(dict_users_unlabeled), '*',len(dict_users_unlabeled[0]))

        net_glob = CNNCifar(args=args).to(args.device)

        print("\n Begin Train")

        net_glob.train()
        w_glob = net_glob.state_dict()

        w_best = copy.deepcopy(w_glob)
        best_loss_valid = 1e10
        loss_server_list = []
        loss_train = []
        loss_test = []
        acc_test = []
        f1_test = []
        auc_test = []
        time_list = []

        val_acc_list, net_list = [], []

        model_h1 = copy.deepcopy(net_glob).to(args.device)
        model_h2 = copy.deepcopy(net_glob).to(args.device)

        for iter in range(args.epochs):
            net_glob.train()
            torch.cuda.empty_cache()
            server = ServerUpdate_fedavg(
                args, 
                dataset_train,
                dict_users_labeled
                )
            w, loss_server = server.train(copy.deepcopy(net_glob).to(args.device))
            loss_server_list.append(loss_server)
            net_glob.load_state_dict(w)

            start_time = time()
            # clients training 
            net_glob.train()

            w_locals, loss_locals = [], []
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

            for idx in idxs_users:
                local = ClientUpdate_fedmatch(
                    args, 
                    dataset_train,
                    dataset_train_weak,
                    dict_users_unlabeled[idx]
                    )
                w, loss = local.train(copy.deepcopy(net_glob).to(args.device), model_h1 = model_h1, model_h2 = model_h2)
                w_locals.append(copy.deepcopy(w)) 
                loss_locals.append(copy.deepcopy(loss))

            model_h1.load_state_dict(w_locals[0])
            model_h2.load_state_dict(w_locals[1])

            end_time = time()
            client_train_time = (end_time - start_time) / m * 1000


            w_glob = FedAvg(w_locals)
            net_glob.load_state_dict(w_glob)
            net_glob.eval()
            
            acc_valid, f1_valid, loss_valid, auc_valid = test(net_glob, dataset_test, args)
            f1_test.append(f1_valid)
            loss_test.append(loss_valid)
            acc_test.append(acc_valid)
            auc_test.append(auc_valid)
            if loss_valid <= best_loss_valid:
                best_loss_valid = loss_valid
                w_best = copy.deepcopy(w_glob)

            loss_avg = sum(loss_locals) / len(loss_locals)
            time_list.append(client_train_time)
            print('Epoch {:3d} | Server-side: loss {:.3f} | Client-side: loss {:.3f}, average time {:.1f}ms | Valid Set: Accuracy {:.4f}, F1 score {:.2f}'.format(iter, loss_server, loss_avg, client_train_time, acc_valid, f1_valid))
            loss_train.append(loss_avg) 
            # print('Round {:3d}, acc_valid {:.2f}%'.format(iter, acc_valid))

        for i in range(args.epochs):
            avg_time_list[i] += time_list[i]
            avg_loss_server_list[i] += loss_server_list[i]
            avg_loss_train[i] += loss_train[i]
            avg_acc_test[i] += acc_test[i]
            avg_f1_test[i] += f1_test[i]
            # avg_auc_test[i] += auc_test[i]


    for i in range(args.epochs):
        avg_time_list[i] /= args.repeat
        avg_loss_server_list[i] /= args.repeat
        avg_loss_train[i] /= args.repeat
        avg_acc_test[i] /= args.repeat
        avg_f1_test[i] /= args.repeat

    fig = plt.figure(figsize=(20,12))
    gs = GridSpec(2, 3, figure=fig)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[0, 2])
    ax4 = plt.subplot(gs[1, 0])
    ax5 = plt.subplot(gs[1, 1])
    ax6 = plt.subplot(gs[1, 2])

    # ax1.plot(loss_train[1:] + [0.45], label = "line 1")
    ax1.plot(avg_time_list, alpha=0.6)
    ax2.plot(avg_loss_server_list, alpha=0.6)
    ax3.plot(avg_loss_train, alpha=0.6)
    ax4.plot(avg_acc_test, alpha=0.6)
    ax5.plot(avg_f1_test, alpha=0.6)
    # ax4.plot()

    ax1.set_title('CPU Training Time per Clients (ms)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Time')

    ax2.set_title('Server Train loss (per epoch)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')

    ax3.set_title('Client Train loss (per epoch)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')

    ax4.set_title('Test accuracy (per epoch)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')

    ax5.set_title('F1-score (per epoch)')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('F1')


    # plt.show()
    fig.savefig('fedmatch' + '_' + str(args.init_epochs) + '_' + str(args.threshold) + '_' + args.iid + '.png')

    # print("\n Begin test")

    # net_glob.load_state_dict(w_best)
    # net_glob.eval()

    # acc_test, loss_test = test(net_glob, dataset_test, args)
    # print("Testing accuracy: {:.2f}% \n\n".format(acc_test))
    

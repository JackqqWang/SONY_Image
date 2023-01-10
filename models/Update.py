#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.autograd import Variable
import copy
import torch.nn.functional as F

def dist_l2(w, w_ema):
    a = 0.0
    for i in list(w.keys()) :
        a += (w[i]-w_ema[i]).float().norm(2)
    return a
                
def dist_l1(w):
    a = 0.0
    for i in list(w.keys()) :
        a +=  w[i].float().norm(1)
    return a

def softmax_mse_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes


def penalty(model,copy_):

    w_epoch = list(model.parameters())
    copy_list=list(copy_.parameters())
    # print("layer0 :", w_epoch[0])
    # print("layer2:",w_epoch[2])
    deltaw_list = []
    k=[]
    # sum_w = 0
    for p in range(len(w_epoch)):
        deltaw_list.append(w_epoch[p]-copy_list[p])
        # print("i:",p,"layer:",deltaw_list[p].flatten())
        c=torch.dot(deltaw_list[p].flatten(), deltaw_list[p].flatten()).reshape(-1)[0]
        # print("c",c)
        if p==0:
            k.append(c)
        else:
            k.append(c+k[p-1])
            # print("sum_list", k)
    sum_w=k[p]
    return sum_w

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

class DatasetSplit_aug(Dataset):
    def __init__(self, dataset, dataset_weak, idxs):
        self.dataset = dataset
        self.dataset_weak = dataset_weak
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        batch = [self.dataset[self.idxs[item]], self.dataset_weak[self.idxs[item]]]
        return batch


def batch_generator(x_mat, y, batch_size, seq_len):
    mean_data_x = np.mean(x_mat, axis=0)
    mean_data_y = np.mean(y, axis=0)
    # pad the beginning of the data with mean rows in order to minimize the error
    # on first rows while using sequence
    prefix_padding_x = np.asarray([mean_data_x for _ in range(seq_len - 1)])
    prefix_padding_y = np.asarray([mean_data_y for _ in range(seq_len - 1)])
    padded_data_x = np.vstack((prefix_padding_x, x_mat))
    padded_data_y = np.vstack((prefix_padding_y, y))
    seq_data = []
    seq_y = []
    for i in range(len(padded_data_x) - seq_len + 1):
        seq_data.append(padded_data_x[i:i + seq_len, :])
        # seq_y.append(padded_data_y[i + seq_len - 1:i + seq_len, :])
        seq_y.append(padded_data_y[i + seq_len - 2:i + seq_len - 1, :])
        if len(seq_data) == batch_size:
            if torch.cuda.is_available():
                yield Variable(torch.cuda.FloatTensor(seq_data)), Variable(torch.cuda.LongTensor(seq_y))
            else:
                yield Variable(torch.FloatTensor(seq_data)), Variable(torch.LongTensor(seq_y))
            seq_data = []
            seq_y = []
    if len(seq_data) > 0:  # handle data which is not multiply of batch size
        if torch.cuda.is_available():
            yield Variable(torch.cuda.FloatTensor(seq_data)), Variable(torch.cuda.LongTensor(seq_y))
        else:
            yield Variable(torch.FloatTensor(seq_data)), Variable(torch.LongTensor(seq_y))

class ServerUpdate_fedavg(object):
    def __init__(self, args, dataset_train, dict_users_labeled):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss(ignore_index= -1)
        self.dict_users_labeled = dict_users_labeled
        self.ldr_train = DataLoader(
            DatasetSplit(dataset = dataset_train, idxs = dict_users_labeled),
            batch_size=self.args.local_bs, 
            shuffle=True
            )

    def train(self, model):
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        # print('Start model training')

        # for epoch in range(self.args.local_ep):
        loss_total = []
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            opt.zero_grad()
            output = model(images)
            loss = self.loss_func(output, labels)
            loss.backward()
            opt.step()
            loss_total.append(loss.item())
        return model.state_dict(), sum(loss_total) / len(loss_total)



class ClientUpdate_fedavg(object):
    def __init__(self, args, dataset_train, dict_users_unlabeled, pseudo_label):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss(ignore_index= -1)
        self.dict_users_unlabeled = dict_users_unlabeled
        self.pseudo_label = pseudo_label
        self.ldr_train = DataLoader(
            DatasetSplit(dataset = dataset_train, idxs = dict_users_unlabeled, pseudo_label = pseudo_label),
            batch_size=self.args.local_bs, 
            shuffle=True
            )


    def train(self, model):
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        # print('Start model training')

        for epoch in range(self.args.local_ep):
            loss_total = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                opt.zero_grad()
                output = model(images)
                loss = self.loss_func(output, labels)
                loss.backward()
                opt.step()
                loss_total.append(loss.item())
        return model.state_dict(), sum(loss_total) / len(loss_total)

        # for epoch in range(self.args.local_ep):
        #     loss_total = []
        #     for i, ((x_batch_pos, y_batch_pos), (x_batch_neg, y_batch_neg)) in enumerate(zip(batch_generator(self.X_train_pos_server, self.y_train_pos_server, self.args.local_bs, 2), batch_generator(self.X_train_neg_server, self.y_train_neg_server, self.args.local_bs, 2))):
        #         # feed data from two dataloader
        #         y_batch_pos = y_batch_pos.to(self.args.device)
        #         y_batch_neg = y_batch_neg.to(self.args.device)
        #         # print(y_batch_pos)
        #         opt.zero_grad()
        #         out_pos = model(x_batch_pos)
        #         out_neg = model(x_batch_neg)
        #         loss_pos = self.loss_func(out_pos, torch.flatten(y_batch_pos))
        #         loss_neg = self.loss_func(out_neg, torch.flatten(y_batch_neg))
        #         loss = loss_pos + loss_neg
        #         loss.backward()
        #         # print(loss.item())
        #         if loss.item() > 0:
        #             loss_total.append(loss.item())
        #         opt.step()
        # return model.state_dict(), sum(loss_total) / len(loss_total) if len(loss_total) != 0 else 0


class ClientUpdate_fedmatch(object):
    def __init__(self, args, dataset_train, dataset_train_weak, dict_users_unlabeled):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss(ignore_index= -1)
        self.dict_users_unlabeled = dict_users_unlabeled
        self.ldr_train = DataLoader(
            DatasetSplit_aug(dataset_train, dataset_train_weak, dict_users_unlabeled),
            batch_size=self.args.local_bs, 
            shuffle=True
            )



    def train(self, model, model_h1, model_h2):
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        model_local_static = copy.deepcopy(model).to(self.args.device)
        # print('Start model training')

        for epoch in range(self.args.local_ep):
            loss_total = []
            for batch_idx, batch in enumerate(self.ldr_train):
                images_1 = batch[0][0].to(self.args.device)
                images_2 = batch[1][0].to(self.args.device)

                opt.zero_grad()
                output = model(images_2)
                output_h1 = model_h1(images_2)
                output_h2 = model_h2(images_2)
                output_aug = model(images_1)

                # _, logits = output.loss, output.logits
                # loss_h1, logits_h1 = output_h1.loss, output_h1.logits
                # loss_h2, logits_h2 = output_h2.loss, output_h2.logits
                # loss_aug, logits_aug = output_aug.loss, output_aug.logits
                logits = output
                logits_h1 = output_h1
                logits_h2 = output_h2
                logits_aug = output_aug


                pseudo_label_1 = torch.softmax(logits.detach_(), dim=-1)
                pseudo_label_2 = torch.softmax(logits_h1.detach_(), dim=-1)
                pseudo_label_3 = torch.softmax(logits_h2.detach_(), dim=-1)

                max_probs1, targets_u1 = torch.max(pseudo_label_1, dim=-1)
                max_probs2, targets_u2 = torch.max(pseudo_label_2, dim=-1)
                max_probs3, targets_u3 = torch.max(pseudo_label_3, dim=-1)

                if torch.equal(targets_u1, targets_u2) and torch.equal(targets_u1, targets_u3):
                    max_probs = torch.max(max_probs1, max_probs2)
                    max_probs = torch.max(max_probs, max_probs3)
                else: 
                    max_probs = max_probs1 - 0.2
                targets_u = targets_u1
                mask = max_probs.ge(0.95).float()
                Lu = (F.cross_entropy(logits_aug, targets_u, reduction='none') * mask).mean()

                lambda_iccs = 0.01
                lambda_l2 = 0.0000001
                lambda_l1 = 0.0000001

                L2 = lambda_l2*dist_l2(model.state_dict(), model_local_static.state_dict())
            
                L3 = lambda_l1*dist_l1(model.state_dict())
            
                loss = lambda_iccs*(Lu) + L2 + L3

                loss.backward()
                opt.step()
                loss_total.append(loss.item())
        return model.state_dict(), sum(loss_total) / len(loss_total)




class ClientUpdate_fedmix(object):
    def __init__(self, args, dataset_train, dataset_train_aug, dict_users_unlabeled):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss(ignore_index= -1)
        self.dict_users_unlabeled = dict_users_unlabeled
        self.ldr_train = DataLoader(
            DatasetSplit_aug(dataset = dataset_train, dataset_weak = dataset_train_aug, idxs = dict_users_unlabeled),
            batch_size=self.args.local_bs, 
            shuffle=True
            )

    def train(self, model, model_2):
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        consistency_criterion = softmax_mse_loss
        # print('Start model training')

        for epoch in range(self.args.local_ep):
            loss_total = []
            for batch_idx, batch in enumerate(self.ldr_train):
                images_1 = batch[0][0].to(self.args.device)
                images_2 = batch[1][0].to(self.args.device)

                opt.zero_grad()
                output = model(images_2)
                output_aug = model(images_1)

                logits = output
                logits_aug = output_aug

                pseudo_label_1 = torch.softmax(logits.detach_(), dim=-1)
                pseudo_label_2 = torch.softmax(logits_aug.detach_(), dim=-1)

                max_probs, targets_u = torch.max((pseudo_label_1 + pseudo_label_2)/2, dim=-1)

                mask = max_probs.ge(0.8).float()
                loss1 = (F.cross_entropy(logits, targets_u, reduction='none') * mask).mean()
                loss2 = consistency_criterion(logits, logits_aug)
                loss3 = penalty(model, model_2)

                loss = loss1 + loss2 + 0.1 * loss3

                loss.backward()
                opt.step()
                loss_total.append(loss.item())
        return model.state_dict(), sum(loss_total) / len(loss_total)


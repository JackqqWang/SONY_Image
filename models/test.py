#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.Update import batch_generator
# import numpy
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def test(net_g, datatest, args):
    net_g.eval()
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    test_trues = []
    test_preds = []

    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum',ignore_index=-1).item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        test_outputs = log_probs.argmax(dim=1)
        # test_outputs.detach().cpu().numpy()
        test_preds.extend(test_outputs.detach().cpu().numpy())
        # test_preds.extend(log_probs[:,1].cpu().detach().numpy())
        test_trues.extend(target.detach().cpu().numpy())
        # test_trues.extend(torch.flatten(target).detach().cpu().numpy())
        
    # print(test_preds[:10])
    # print(test_trues[:10])

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    sklearn_f1 = f1_score(test_trues, test_preds, average='micro')
    # sklearn_auc = roc_auc_score(test_trues, test_preds, multi_class='ovr', average='micro')
    return accuracy, sklearn_f1, test_loss, 0


# def test(net_g, dataset, args):
#     net_g.eval()
#     test_loss = 0
#     TP, TN, total, total_batch = 0, 0, 0, 0
#     loss_func = nn.CrossEntropyLoss(ignore_index= -1)
#     label_all = []
#     prob_all = []

#     data_loader = DataLoader(dataset, batch_size=args.bs)
#     for idx, (data, target) in enumerate(data_loader):
#         if args.gpu != -1:
#             data, target = data.cuda(), target.cuda()
#         log_probs = net_g(data)
#         test_loss += F.cross_entropy(log_probs, target, reduction='sum',ignore_index=-1).item()
#         y_pred = log_probs.data.max(1, keepdim=True)[1]
#         correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()



#     # eval on test_pos
#     for i, (x_batch, y_batch) in enumerate(batch_generator(X_test_pos, y_test_pos, args.local_bs, 2)):
#         y_batch = y_batch.to(args.device)
#         out = model(x_batch)
#         prob_all.extend(out[:,1].cpu().detach().numpy())
#         label_all.extend(torch.flatten(y_batch))
#         test_loss += loss_func(out, torch.flatten(y_batch)).item()
#         out = out.squeeze(dim = 0)
#         preds = F.log_softmax(out, dim=1).argmax(dim=1)
#         total += y_batch.size(0)
#         total_batch += 1
#         TP += (preds == torch.flatten(y_batch)).sum().item()

#     # eval on test_neg
#     for i, (x_batch, y_batch) in enumerate(batch_generator(X_test_neg, y_test_neg, args.local_bs, 2)):
#         y_batch = y_batch.to(args.device)
#         out = model(x_batch)
#         prob_all.extend(out[:,1].cpu().detach().numpy())
#         label_all.extend(torch.flatten(y_batch))
#         test_loss += loss_func(out, torch.flatten(y_batch)).item()
#         out = out.squeeze(dim = 0)
#         preds = F.log_softmax(out, dim=1).argmax(dim=1)
#         total += y_batch.size(0)
#         total_batch += 1
#         TN += (preds == torch.flatten(y_batch)).sum().item()

#     acc_test = (TP + TN) / total
#     F1 = TP / (TP + (total - TP - TN) / 2)
#     auc = roc_auc_score(label_all, prob_all)
    
#     return acc_test, F1, test_loss / total_batch, auc
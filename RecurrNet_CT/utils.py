import os
import time
import logging
import numpy as np
import configparser
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored


def c_index(risk_pred, y, e):
    ''' Performs calculating c-index

    :param risk_pred: (np.ndarray or torch.Tensor) model prediction
    :param y: (np.ndarray or torch.Tensor) the times of event e
    :param e: (np.ndarray or torch.Tensor) flag that records whether the event occurs
    :return c_index: the c_index is calculated by (risk_pred, y, e)
    '''

    # if not isinstance(y, np.ndarray):
    #     y = y.detach().cpu().numpy()
    # if not isinstance(risk_pred, np.ndarray):
    #     risk_pred = risk_pred.detach().cpu().numpy()
    # if not isinstance(e, np.ndarray):
    #     e = e.detach().cpu().numpy()
    # return concordance_index(y, risk_pred, e)

    if not isinstance(y, np.ndarray):
        y = y.squeeze(1).detach().cpu().numpy()
    if not isinstance(risk_pred, np.ndarray):
        risk_pred = -risk_pred.squeeze(1).detach().cpu().numpy()
    if not isinstance(e, np.ndarray):
        e = e.squeeze(1).detach().cpu().numpy().astype(bool)
    c_in = concordance_index_censored(e, y, risk_pred)
    return c_in[0]

class Regularization(object):
    def __init__(self, order, weight_decay):
        ''' The initialization of Regularization class

        :param order: (int) norm order number
        :param weight_decay: (float) weight decay rate
        '''
        super(Regularization, self).__init__()
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model):
        ''' Performs calculates regularization(self.order) loss for model.

        :param model: (torch.nn.Module object)
        :return reg_loss: (torch.Tensor) the regularization(self.order) loss
        '''
        reg_loss = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        reg_loss = self.weight_decay * reg_loss
        return reg_loss

class NLog_Likelihood(nn.Module):
    """Negative partial log-likelihood of Cox proportional hazards model"""

    def __init__(self, L2_reg):
        super(NLog_Likelihood, self).__init__()
        self.reg = Regularization(order=2, weight_decay=L2_reg)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, risk_pred, y, e, model):

        time, indices = torch.sort(y.squeeze(1), 0, descending=True)
        event = e.squeeze(1)[indices]
        xw = risk_pred.squeeze(1)[indices]
        n_samples = xw.shape[0]

        loss  = -torch.sum(xw*event - event*torch.log(torch.exp(xw).cumsum(0))) / n_samples
        # print(loss)
        # loss = torch.log(loss)

        l2_loss = self.reg(model)
        return loss, l2_loss


# class NegativeLogLikelihood(nn.Module):
#     def __init__(self, L2_reg):
#         super(NegativeLogLikelihood, self).__init__()
#         self.L2_reg = L2_reg
#         self.reg = Regularization(order=2, weight_decay=self.L2_reg)
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     def forward(self, risk_pred, y, e, model):
#         mask = torch.ones(y.shape[0], y.shape[0]).to(self.device)
#         # print(mask.shape)
#         mask[(y.T - y) > 0] = 0
#         log_loss = torch.exp(risk_pred) * mask
#         # log_loss = torch.log(torch.sum(log_loss, dim=0)) / torch.sum(mask, dim=0)
#         # log_loss = log_loss.reshape(-1, 1)
#         log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
#         log_loss = torch.log(log_loss).reshape(-1, 1)
#         neg_log_loss = -torch.sum((risk_pred-log_loss) * e) / torch.sum(e)
#         l2_loss = self.reg(model)
#         # print(neg_log_loss)
#         return neg_log_loss, l2_loss

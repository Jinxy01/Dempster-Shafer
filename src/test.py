"""
@author: Tiago Roxo, UBI
@date: 2020
"""
from torch import tensor
from torch.nn.functional import one_hot
from torch.nn import MSELoss as MSE, CrossEntropyLoss as CE

from itertools import chain, combinations
import torch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from utils.dempster_shaffer import *
from utils.aux_function import *

# ----------------------------
def cross_entropy_one_hot(y_hat, y):
    _, y_labels = y.max(dim=0)
    return CE()(y_hat, y_labels) # Não dá zero quando são iguais... mas é o mínimo


def test():
    # Testing
    Y_Train = tensor([1,0])
    y_hat = tensor([1,0])
    y = one_hot(Y_Train, num_classes=2).float()
    y_hat = one_hot(y_hat, num_classes=2).float()
    loss = MSE()
    output = loss(y, y_hat)
    print(output)
    #output = loss(y, y_hat)
    print(cross_entropy_one_hot(y_hat, y))

# ----------------------------

# def model_predict(x,y, rule_set):
#     M = []
#     for m,s in rule_set:
#         if s(x,y): # Point coordinates (y is NOT label class here)
#             M.append(m)

#     m = (0.,0.,1) # Full uncertainty
#     for m_i in M:
#         m = dempster_rule(m,m_i)
#     print(m)
#     print(M)


def model_predict(x,y, rule_set):
    M = []
    for m,s in rule_set:
        if s(x,y): # Point coordinates (y is NOT label class here)
            M.append(m)

    m = weight_full_uncertainty()
    for m_i in M:
        m = dempster_rule(m,m_i)

    y_hat = y_argmax(belief(m))
    return frozenset_to_class(y_hat)


# def start_weights(s_list):
#     list_initial_weights = []
#     for s in s_list:
#         r   = torch.tensor(0.04, device=DEVICE, dtype=DTYPE, requires_grad=True)
#         b   = torch.tensor(0.06, device=DEVICE, dtype=DTYPE, requires_grad=True)
#         r_b = torch.tensor(0.9, device=DEVICE, dtype=DTYPE, requires_grad=True)
#         list_initial_weights.append([(r,b,r_b), s])
#     return list_initial_weights
        



if __name__ == "__main__":

    Y_Train = tensor([1,0])
    X      = [(0.2, 0.2)], (0.3, -0.4)]
    s_list = [lambda x,y: y > 0, lambda x,y: y <= 0]
    #rule_set = start_weights(s_list)

    rule_set = start_weights(s_list)

    for (x,y) in X:
        print(x,y)
        print(model_predict(x,y,rule_set))

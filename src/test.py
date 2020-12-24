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

# def model_predict(x,y, rule_set):
#     M = []
#     for m,s in rule_set:
#         if s(x,y): # Point coordinates (y is NOT label class here)
#             M.append(m)

#     m = weight_full_uncertainty()
#     for m_i in M:
#         m = dempster_rule(m,m_i)

#     y_hat = y_argmax(belief(m))
#     return frozenset_to_class(y_hat)


def model_predict_train(x,y, rule_set):
    M = []
    for m,_,s in rule_set:
        if s(x,y): # Point coordinates (y is NOT label class here)
            M.append(m)

    m = weight_full_uncertainty()
    for m_i in M:
        m = dempster_rule(m,m_i)

    # y_hat = frozenset_to_class(y_argmax(belief(m)))
    # y_hat_one_hot = one_hot(tensor(y_hat), num_classes=NUM_CLASSES).float()
    #
    y_hat_prob = y_argmax_train(plausibility(m))

    # # Not working...
    # print(y_hat_prob, y_hat_one_hot)
    # y_hat_prob_tensor = torch.tensor(y_hat_prob)
    # print(y_hat_prob_tensor, y_hat_one_hot)
    # y_hat = y_hat_prob[0] * y_hat_one_hot
    # print(y_hat)
    # #exit(0)
    # #print(y_hat)
    return y_hat_prob

def model_predict_train_v2(x,y, rule_set):
    M = []
    for m,_,s in rule_set:
        if s(x,y): # Point coordinates (y is NOT label class here)
            M.append(m)

    m = weight_full_uncertainty()
    for m_i in M:
        m = dempster_rule(m,m_i)

    # y_hat = frozenset_to_class(y_argmax(belief(m)))
    # y_hat_one_hot = one_hot(tensor(y_hat), num_classes=NUM_CLASSES).float()
    #
    y_hat_prob = y_argmax_train_v2(m)

    # # Not working...
    # print(y_hat_prob, y_hat_one_hot)
    # y_hat_prob_tensor = torch.tensor(y_hat_prob)
    # print(y_hat_prob_tensor, y_hat_one_hot)
    # y_hat = y_hat_prob[0] * y_hat_one_hot
    # print(y_hat)
    # #exit(0)
    # #print(y_hat)
    return y_hat_prob

def optimization(X, Y, rule_set, loss):

    for t in range(50):
        y_hat_list = []
        for x,y in X:
            y_hat = model_predict_train_v2(x,y, rule_set)
            y_hat_list.append(y_hat)
        
        # Convert to one hot encoder
        batch_loss = mse(Y, y_hat_list)
        #exit(0)

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model).
        for m, optim, s in rule_set:
            optim.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        batch_loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        for m, optim, s in rule_set:
            optim.step()

        # Projection to respect Dempster Shaffer conditions
        # Page 47
        #project_masses(rule_set)

        if t % 10 == 0:
            print(t, batch_loss.item())

    print(rule_set)

# def start_weights(s_list):
#     list_initial_weights = []
#     for s in s_list:
#         r   = torch.tensor(0.04, device=DEVICE, dtype=DTYPE, requires_grad=True)
#         b   = torch.tensor(0.06, device=DEVICE, dtype=DTYPE, requires_grad=True)
#         r_b = torch.tensor(0.9, device=DEVICE, dtype=DTYPE, requires_grad=True)
#         list_initial_weights.append([(r,b,r_b), s])
#     return list_initial_weights
        
# def mse(y, y_hat):
#     # Y_hat is the predicted one
#     list_loss = []
#     tot = len(y)
#     for i in range(tot):
#         y0, y1 = y[i]
#         y_hat0, y_hat1 = y_hat[i]
#         y0_loss = (y0 - y_hat0).pow(2)
#         y1_loss = (y1 - y_hat1).pow(2)
#         list_loss.append([y0_loss/(2*tot), y1_loss/(2*tot)])
    
#     return list_loss

def mse(y, y_hat):
    # Y_hat is the predicted one
    sum_ = 0.
    tot = len(y)
    for i in range(tot):
        y0, y1 = y[i]
        y_hat0, y_hat1 = y_hat[i]
        y0_loss = (y0 - y_hat0).pow(2)
        y1_loss = (y1 - y_hat1).pow(2)
        sum_ += y0_loss + y1_loss
    
    return sum_/(2*tot)


if __name__ == "__main__":

    Y_Train = tensor([1,0])
    Y = one_hot(Y_Train, num_classes=NUM_CLASSES).float()

    X      = [(0.2, 0.2), (0.3, -0.4)]
    s_list = [lambda x,y: y > 0, lambda x,y: y <= 0]#, lambda x,y: x != 0]
    loss = MSE()
    #rule_set = start_weights(s_list)

    rule_set = start_weights(s_list)
    #print(rule_set)

    #for (x,y) in X:
    #    print(x,y)
    #    print(model_predict_train(x,y,rule_set))
    optimization(X, Y, rule_set, loss)


"""
@author: Tiago Roxo, UBI
@date: 2020
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import *
from utils.aux_function import *

def is_converged(loss_current, loss_previous):
    convergence = abs(loss_current-loss_previous) <= EPSILON
    #print(np.size(convergence) - np.count_nonzero(convergence))
    # All rules have converged to minimal loss
    return convergence.item()

def y_argmax_train_v2(dict_m):
    r, b, r_b = dict_m[frozenset({'R'})], dict_m[frozenset({'B'})], dict_m[frozenset({'R', 'B'})] 
    #max_m = max(r, b)
    p_a = r + r_b
    p_b = b + r_b
    p_tot = p_a + p_b
    return p_a/p_tot, p_b/p_tot, 1 # It works with projection working!
    #return r, b, r_b
    #return r/(r+r_b), b/(r+r_b), r_b 
    #return r/(r+b+r_b), b/(r+b+r_b), r_b 
    #return (r+p_a)/2, (b+p_b)/2, r_b # Uncertainty 1.0 with useless rule


def model_predict_train_v2(x,y, rule_set):
    M = []
    for m,_,s in rule_set:
        if s(x,y): # Point coordinates (y is NOT label class here)
            M.append(m)

    m = weight_full_uncertainty()
    for m_i in M:
        m = dempster_rule(m,m_i)
        
    r_prob, b_prob, uncertainty = y_argmax_train_v2(m)
    y_hat = [r_prob, b_prob]

    return y_hat


def training(X, Y, rule_set, loss):

    previous_loss = sys.maxsize

    for t in range(1000):
        y_hat_list = []
        for x,y in X:
            y_hat = model_predict_train_v2(x,y, rule_set)
            y_hat_list.append(y_hat)
        
        # Convert to one hot encoder
        batch_loss = mse(Y, y_hat_list)

        if (is_converged(batch_loss, previous_loss)):
            print("Breaking at {} iteration".format(t))
            break

        previous_loss = batch_loss

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model).
        for _, optim, s in rule_set:
            optim.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        batch_loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        for _, optim, s in rule_set:
            optim.step()

            # Projection
            for p in optim.param_groups[0]['params']:
                p.data.clamp_(min=0, max=1)

        if t % 10 == 0:
            print(t, batch_loss.item())
        

    normalize_rule_set(rule_set)
    read_rules(rule_set)

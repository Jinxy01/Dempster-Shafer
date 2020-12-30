
"""
@author: Tiago Roxo, UBI
@date: 2020
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import *
from utils.aux_function import *
from utils.dempster_shaffer import *
from utils.a1_helper import *
from utils.bc_helper import *

# ---------------- Training -------------------

def is_converged(loss_current, loss_previous):
    convergence = abs(loss_current-loss_previous) <= EPSILON
    #print(np.size(convergence) - np.count_nonzero(convergence))
    # All rules have converged to minimal loss
    return convergence.item()

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
    
    return sum_/(NUM_CLASSES*tot)

def get_class_probabilities_a1(dict_m):
    r, b, r_b = dict_m[frozenset({'R'})], dict_m[frozenset({'B'})], dict_m[frozenset({'R', 'B'})] 
    #max_m = max(r, b)
    p_a = r + r_b
    p_b = b + r_b
    p_tot = p_a + p_b
    return p_a/p_tot, p_b/p_tot # It works with projection working!
    #return r, b, r_b
    #return r/(r+r_b), b/(r+r_b), r_b 
    #return r/(r+b+r_b), b/(r+b+r_b), r_b 
    #return (r+p_a)/2, (b+p_b)/2, r_b # Uncertainty 1.0 with useless rule

def get_class_probabilities_bc(dict_m):
    b   = dict_m[frozenset({'B'})]
    m   = dict_m[frozenset({'M'})]
    b_m = dict_m[frozenset({'B','M'})]
    #max_m = max(r, b)
    p_b = b + b_m
    p_m = m + b_m
    p_tot = p_b + p_m
    return p_m/p_tot, p_b/p_tot # It works with projection working!



def model_predict_train_a1(x,y, rule_set):
    M = []
    for m,_,s in rule_set:
        if s(x,y): # Point coordinates (y is NOT label class here)
            M.append(m)

    m = weight_full_uncertainty_A1()
    for m_i in M:
        m = dempster_rule(m,m_i)
        
    r_prob, b_prob = get_class_probabilities_a1(m)
    # Change order to match one hot encoding of classes
    # Blue (class 0) => [1 0]
    # Red  (class 1) => [0 1]
    y_hat = [b_prob, r_prob]

    return y_hat

def model_predict_train_bc(a,b,c,d,e,f,g,h,i, rule_set):
    M = []
    for m,_,s in rule_set:
        if s(a,b,c,d,e,f,g,h,i): # Point coordinates (y is NOT label class here)
            M.append(m)

    m = weight_full_uncertainty_bc()
    for m_i in M:
        m = dempster_rule(m,m_i)
        
    r_prob, b_prob = get_class_probabilities_bc(m)
    # Change order to match one hot encoding of classes
    # Blue (class 0) => [1 0]
    # Red  (class 1) => [0 1]
    y_hat = [b_prob, r_prob]

    return y_hat


def model_predict_train_test(X, rule_set, dataset_name):
    if dataset_name == "A1_Dataset":
        y_hat_list = []
        for x,y in X:
            y_hat = model_predict_train_a1(x,y, rule_set)
            y_hat_list.append(y_hat)
        return y_hat_list
    elif dataset_name == "BC_Dataset":
        y_hat_list = []
        for a,b,c,d,e,f,g,h,i in X:
            y_hat = model_predict_train_bc(a,b,c,d,e,f,g,h,i,rule_set)
            y_hat_list.append(y_hat)
        return y_hat_list
    else:
        assert False


def training(X, Y, rule_set, loss, dataset_name):

    it_loss = []
    previous_loss = sys.maxsize

    for i in range(50):
        # Model predictions
        y_hat_list = model_predict_train_test(X, rule_set, dataset_name)

        # Compute loss
        batch_loss = mse(Y, y_hat_list)
        it_loss.append(batch_loss)

        if (is_converged(batch_loss, previous_loss)):
            print(BREAK_IT.format(i))
            break

        previous_loss = batch_loss

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model).
        for _, optim, _ in rule_set:
            optim.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        batch_loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        for _, optim, _ in rule_set:
            optim.step()

            # Projection
            for p in optim.param_groups[0]['params']:
                p.data.clamp_(min=0, max=1)

        if i % 10 == 0:
            print(i, batch_loss.item())

    normalize_rule_set(rule_set)
    return rule_set, it_loss


# ---------------- Inference -------------------

def frozenset_to_class(y_hat):
    # For a1 and a2 dataset
    #if y_hat == frozenset({'R', 'B'}):
    #    assert False
    if y_hat == frozenset({'R'}):
        return 1 # Red is class 1
    return 0

def y_argmax(dict_m):
    return frozenset_to_class(max(dict_m, key=(lambda key: dict_m[key])))


def model_inference_a1(x,y, rule_set, dataset_name):
    M = []
    for m,_,s in rule_set:
        if s(x,y): # Point coordinates (y is NOT label class here)
            M.append(m)

    m = weight_full_uncertainty_A1()
    for m_i in M:
        m = dempster_rule(m,m_i)
        
    return y_argmax(m)

def model_inference_bc(a,b,c,d,e,f,g,h,i, rule_set):
    M = []
    for m,_,s in rule_set:
        if s(a,b,c,d,e,f,g,h,i): # Point coordinates (y is NOT label class here)
            M.append(m)

    m = weight_full_uncertainty_A1()
    for m_i in M:
        m = dempster_rule(m,m_i)
        
    return y_argmax(m)


def inference_test(X, rule_set, dataset_name):
    if dataset_name == "A1_Dataset":
        y_hat_list = []
        for x,y in X:
            y_hat = model_inference_a1(x,y, rule_set, dataset_name)
            y_hat_list.append(y_hat)
        return y_hat_list
    elif dataset_name == "BC_Dataset":
        y_hat_list = []
        for a,b,c,d,e,f,g,h,i in X:
            y_hat = model_inference_bc(a,b,c,d,e,f,g,h,i, rule_set)
            y_hat_list.append(y_hat)
        return y_hat_list
    else:
        assert False

def inference(X, Y, rule_set, dataset_name):
    y_hat_list = inference_test(X, rule_set, dataset_name)
       
    tot_correct_predicts = np.sum(np.array(Y) == np.array(y_hat_list))
    tot_predicts = len(Y)
    return tot_correct_predicts/tot_predicts*100, tot_correct_predicts, tot_predicts
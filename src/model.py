
"""
@author: Tiago Roxo, UBI
@date: 2020
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import *
from utils.common import *
from utils.dempster_shaffer import *
from utils.bc_helper import *

#from torch.utils.data import DataLoader
from statistics import mean


# ---------------- Training -------------------

def is_converged(loss_current, loss_previous):
    convergence = abs(loss_current-loss_previous) <= EPSILON
    return convergence

def get_two_class_probabilities(dict_m, dataset_name):
    class_0, class_1 = get_class_plausibility(plausibility(dict_m, dataset_name), dataset_name)
    prob_class_0 = class_0/(class_0+class_1)
    prob_class_1 = class_1/(class_0+class_1)
    return prob_class_0, prob_class_1

def get_three_class_probabilities(dict_m, dataset_name):
    class_0, class_1, class_2 = get_class_plausibility(plausibility(dict_m, dataset_name), dataset_name)
    prob_class_0 = class_0/(class_0+class_1+class_2)
    prob_class_1 = class_1/(class_0+class_1+class_2)
    prob_class_2 = class_2/(class_0+class_1+class_2)
    return prob_class_0, prob_class_1, prob_class_2

# def get_ten_class_probabilities(dict_m, dataset_name):
#     class_0, class_1, class_2, class_3, class_4, class_5, class_6, class_7, class_8, class_9 = get_class_plausibility(plausibility(dict_m, dataset_name), dataset_name)
#     prob_class_0 = class_0/(class_0+class_1+class_2+class_3+class_4+class_5+class_6+class_7+class_8+class_9)
#     prob_class_1 = class_1/(class_0+class_1+class_2+class_3+class_4+class_5+class_6+class_7+class_8+class_9)
#     prob_class_2 = class_2/(class_0+class_1+class_2+class_3+class_4+class_5+class_6+class_7+class_8+class_9)
#     prob_class_3 = class_3/(class_0+class_1+class_2+class_3+class_4+class_5+class_6+class_7+class_8+class_9)
#     prob_class_4 = class_4/(class_0+class_1+class_2+class_3+class_4+class_5+class_6+class_7+class_8+class_9)
#     prob_class_5 = class_5/(class_0+class_1+class_2+class_3+class_4+class_5+class_6+class_7+class_8+class_9)
#     prob_class_6 = class_6/(class_0+class_1+class_2+class_3+class_4+class_5+class_6+class_7+class_8+class_9)
#     prob_class_7 = class_7/(class_0+class_1+class_2+class_3+class_4+class_5+class_6+class_7+class_8+class_9)
#     prob_class_8 = class_8/(class_0+class_1+class_2+class_3+class_4+class_5+class_6+class_7+class_8+class_9)
#     prob_class_9 = class_9/(class_0+class_1+class_2+class_3+class_4+class_5+class_6+class_7+class_8+class_9)
#     return prob_class_0, prob_class_1, prob_class_2, prob_class_3, prob_class_4, prob_class_5, prob_class_6, prob_class_7, prob_class_8, prob_class_9


def prediction(rule_set, dataset_name, *att):
    # Args is x,y in A1 and 9 attributes in Breast Cancer
    M = []
    for i,(m,_,s) in enumerate(rule_set):
        # if i+1 not in ALLOWED_RULES: # Only look at allowed rules
        #     continue
        if s(*att): # Point coordinates (y is NOT label class here)
            M.append(m)

    # m = weight_full_uncertainty(dataset_name)
    # for m_i in M:
    #     m = dempster_rule(m,m_i, dataset_name)

    m = M[0]
    for i in range(1,len(M)):
        m = dempster_rule(m,M[i], dataset_name)

    if dataset_name == "IRIS_Dataset" or dataset_name == "WINE_Dataset": # Has 3 classes
        prob_class_0, prob_class_1, prob_class_2 = get_three_class_probabilities(m, dataset_name)
        p_0 = prob_class_0 * CLASS_0_ONE_HOT
        p_1 = prob_class_1 * CLASS_1_ONE_HOT
        p_2 = prob_class_2 * CLASS_2_ONE_HOT
        return torch.sum(torch.stack([p_0, p_1, p_2]), dim=0) # Probabilities for three classes

    # elif dataset_name == "DIG_Dataset":
    #     prob_class_0, prob_class_1, prob_class_2, prob_class_3, prob_class_4, prob_class_5, prob_class_6, prob_class_7, prob_class_8, prob_class_9 = get_ten_class_probabilities(m, dataset_name)
    #     p_0 = prob_class_0 * CLASS_0_ONE_HOT
    #     p_1 = prob_class_1 * CLASS_1_ONE_HOT
    #     p_2 = prob_class_2 * CLASS_2_ONE_HOT
    #     p_3 = prob_class_3 * CLASS_0_ONE_HOT
    #     p_4 = prob_class_4 * CLASS_1_ONE_HOT
    #     p_5 = prob_class_5 * CLASS_2_ONE_HOT
    #     p_6 = prob_class_6 * CLASS_0_ONE_HOT
    #     p_7 = prob_class_7 * CLASS_1_ONE_HOT
    #     p_8 = prob_class_8 * CLASS_2_ONE_HOT
    #     p_9 = prob_class_9 * CLASS_2_ONE_HOT
    #     return torch.sum(torch.stack([p_0, p_1, p_2, p_3, p_4, p_5,p_6, p_7, p_8,p_9]), dim=0) # Probabilities for both classes

    else:
        prob_class_0, prob_class_1 = get_two_class_probabilities(m, dataset_name)
        p_0 = prob_class_0 * CLASS_0_ONE_HOT
        p_1 = prob_class_1 * CLASS_1_ONE_HOT
        return torch.sum(torch.stack([p_0, p_1]), dim=0) # Probabilities for both classes
    
def model_predict(X, rule_set, dataset_name):
    y_hat_list = []
    for att in X:
        y_hat = prediction(rule_set, dataset_name, *att)
        y_hat_list.append(y_hat)
    return y_hat_list

# Batching
def batch(lst, i, n):
    for j in range(i, len(lst), n):
        return lst[j:j + n]

def training(X, Y, rule_set, loss, dataset_name):

    training_loss = []
    previous_loss = sys.maxsize
    # Batch info
    batch_size = BATCH_SIZE
    print("Batch size =", batch_size)
    tot = int(len(X)/batch_size)

    for t in range(NUM_EPOCHS):
        epoch_loss = []
        
        for i in range(tot):
            X_batch = batch(X,i*batch_size,batch_size)
            Y_batch = batch(Y,i*batch_size,batch_size)
            #X_batch = X
            #Y_batch = Y

            # For cuda purposes
            Y_batch = Y_batch.to(device=DEVICE)
            # Model predictions
            y_hat_list = model_predict(X_batch, rule_set, dataset_name)

            # Compute loss
            y_hat_list = torch.stack(y_hat_list)
            
            batch_loss = loss(Y_batch, y_hat_list)
            epoch_loss.append(batch_loss.item())

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

        current_epoch_loss = mean(epoch_loss)
        training_loss.append(current_epoch_loss)
        if (is_converged(current_epoch_loss, previous_loss)):
            print(BREAK_IT.format(t))
            break

        previous_loss = current_epoch_loss

        if t % 10 == 0:
            print(t,current_epoch_loss)

    normalize_rule_set(rule_set)
    return rule_set, training_loss


# ---------------- Inference -------------------

def frozenset_to_class(y_hat, dataset_name):
    # For a1 and a2 dataset
    #if y_hat == frozenset({'R', 'B'}):
    #    assert False
    if dataset_name == "A1_Dataset":
        if y_hat == frozenset({'R'}):
            return 1 # Red is class 1
        return 0
    elif dataset_name == "BC_Dataset":
        if y_hat == frozenset({'M'}):
            return 1 # Malign is class 1
        return 0
    elif dataset_name == "IRIS_Dataset":
        if y_hat == frozenset({'S'}):
            return 0 
        if y_hat == frozenset({'C'}):
            return 1 
        return 2 # V
    elif dataset_name == "HD_Dataset":
        if y_hat == frozenset({'A'}):
            return 0 
        return 1 # P
    elif dataset_name == "WINE_Dataset":
        if y_hat == frozenset({'A'}):
            return 0 
        if y_hat == frozenset({'B'}):
            return 1 
        return 2 # C
    elif dataset_name == "DIG_Dataset":
        if y_hat == frozenset({'0'}):
            return 0 
        return 1 # 1
        # if y_hat == frozenset({'2'}):
        #     return 2 
        # if y_hat == frozenset({'3'}):
        #     return 3 
        # if y_hat == frozenset({'4'}):
        #     return 4
        # if y_hat == frozenset({'5'}):
        #     return 5
        # if y_hat == frozenset({'6'}):
        #     return 6 
        # if y_hat == frozenset({'7'}):
        #     return 7 
        # if y_hat == frozenset({'8'}):
        #     return 8 
        # return 9 # 9
    else:
        assert False

def y_argmax(dict_m, dataset_name):
    return frozenset_to_class(max(dict_m, key=(lambda key: dict_m[key])), dataset_name)


def model_inference(rule_set, dataset_name, *att):
    M = []
    for i,(m,_,s) in enumerate(rule_set):
        # if i+1 not in ALLOWED_RULES: # Only look at allowed rules
        #     continue
        if s(*att): # Point coordinates (y is NOT label class here)
            M.append(m)

    # m = weight_full_uncertainty(dataset_name)
    # for m_i in M:
    #     m = dempster_rule(m,m_i, dataset_name)

    m = M[0]
    for i in range(1,len(M)):
        m = dempster_rule(m,M[i], dataset_name)
        
    return y_argmax(m, dataset_name)


def inference(X, rule_set, dataset_name):
    y_hat_list = []
    for att in X:
        y_hat = model_inference(rule_set, dataset_name, *att)
        y_hat_list.append(y_hat)
    return y_hat_list

def model_evaluation(X, Y, rule_set, dataset_name):
    y_hat_list = inference(X, rule_set, dataset_name)
    tot_correct_predicts = np.sum(np.array(Y) == np.array(y_hat_list))
    tot_predicts = len(Y)
    return tot_correct_predicts/tot_predicts*100, tot_correct_predicts, tot_predicts
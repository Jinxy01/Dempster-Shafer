
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

def mse_test(y, y_hat):
    return torch.sum((y_hat - y) ** 2)

def get_two_class_probabilities(dict_m, dataset_name):
    # r, b, r_b = dict_m[frozenset({'R'})], dict_m[frozenset({'B'})], dict_m[frozenset({'R', 'B'})]
    # #max_m = max(r, b)
    # p_a = r + r_b
    # p_b = b + r_b
    # p_tot = p_a + p_b
    # return p_a/p_tot, p_b/p_tot # It works with projection working!
    class_0, class_1 = get_class_plausibility(plausibility(dict_m, dataset_name), dataset_name)
    prob_class_0 = class_0/(class_0+class_1)
    prob_class_1 = class_1/(class_0+class_1)
    return prob_class_0, prob_class_1
    #return r, b, r_b
    #return r/(r+r_b), b/(r+r_b), r_b 
    #return r/(r+b+r_b), b/(r+b+r_b), r_b 
    #return (r+p_a)/2, (b+p_b)/2, r_b # Uncertainty 1.0 with useless rule


def prediction(rule_set, dataset_name, *att):
    # Args is x,y in A1 and 9 attributes in Breast Cancer
    M = []
    for m,_,s in rule_set:
        if s(*att): # Point coordinates (y is NOT label class here)
            M.append(m)

    # m = weight_full_uncertainty(dataset_name)
    # for m_i in M:
    #     m = dempster_rule(m,m_i, dataset_name)

    m = M[0]
    for i in range(1,len(M)):
        m = dempster_rule(m,M[i], dataset_name)
         
    prob_class_0, prob_class_1 = get_two_class_probabilities(m, dataset_name)
    # Change order to match one hot encoding of classes
    # Blue (class 0) => [1 0]
    # Red  (class 1) => [0 1]
    if prob_class_0 > prob_class_1:
        return prob_class_0 * CLASS_0_ONE_HOT
    return prob_class_1 * CLASS_1_ONE_HOT
    # y_hat = [prob_class_0, prob_class_1]

    # return y_hat

    
def model_predict(X, rule_set, dataset_name):
    y_hat_list = []
    for att in X:
        y_hat = prediction(rule_set, dataset_name, *att)
        y_hat_list.append(y_hat)
    return y_hat_list


def training(X, Y, rule_set, loss, dataset_name):

    it_loss = []
    previous_loss = sys.maxsize

    for i in range(NUM_EPOCHS):
        # Model predictions
        y_hat_list = model_predict(X, rule_set, dataset_name)

        # Compute loss

        # Previous
        #batch_loss = mse(Y, y_hat_list)
 
        y_hat_list = torch.stack(y_hat_list)
        batch_loss = loss(Y, y_hat_list)
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

        if i % 50 == 0:
            print(i, batch_loss.item())

    normalize_rule_set(rule_set)
    return rule_set, it_loss


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
    else:
        assert False

def y_argmax(dict_m, dataset_name):
    return frozenset_to_class(max(dict_m, key=(lambda key: dict_m[key])), dataset_name)


def model_inference(rule_set, dataset_name, *att):
    M = []
    for m,_,s in rule_set:
        if s(*att): # Point coordinates (y is NOT label class here)
            M.append(m)

    # m = weight_full_uncertainty(dataset_name)
    # for m_i in M:
    #     m = dempster_rule(m,m_i, dataset_name)

    m = M[0]
    for i in range(1,len(M)):
        m = dempster_rule(m,M[i], dataset_name)
        
    return y_argmax(m, dataset_name)


def inference_test(X, rule_set, dataset_name):
    y_hat_list = []
    for att in X:
        y_hat = model_inference(rule_set, dataset_name, *att)
        y_hat_list.append(y_hat)
    return y_hat_list

def inference(X, Y, rule_set, dataset_name):
    y_hat_list = inference_test(X, rule_set, dataset_name)
    tot_correct_predicts = np.sum(np.array(Y) == np.array(y_hat_list))
    tot_predicts = len(Y)
    return tot_correct_predicts/tot_predicts*100, tot_correct_predicts, tot_predicts
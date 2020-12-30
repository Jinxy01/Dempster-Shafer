
"""
@author: Tiago Roxo, UBI
@date: 2020
"""

from torch import tensor, optim
from utils.config import *
from utils.dempster_shaffer import *

def y_argmax_train(dict_m):
    # Return probability for each classl using Cobb and Shenoy approach
    list_dict_m = list(dict_m.values())
    e1, e2 = list_dict_m
    list_dict_m[0] = e1/(e1+e2)
    list_dict_m[1] = e2/(e1+e2)
    return list_dict_m


# Translates frozenset to class
def frozenset_to_class(y_hat):
    # For a1 and a2 dataset
    if y_hat == frozenset({'R'}):
        return 1 # Red is class 1
    return 0


def is_converged(loss_current, loss_previous):
    convergence = abs(loss_current-loss_previous) <= EPSILON
    #print(np.size(convergence) - np.count_nonzero(convergence))
    # All rules have converged to minimal loss
    return convergence.item()

# Testing

def normalize_masses_combined(dict_combined_m):
    sum_m = 0
    for _, m in dict_combined_m.items():
        sum_m += m
    
    # It is already normalized
    if sum_m == 1.0:
        return dict_combined_m

    dict_combined_m_norm = {}
    for s in dict_combined_m:
        dict_combined_m_norm[s] = dict_combined_m[s]/sum_m
    
    return dict_combined_m_norm



# ----------------------------------------------

def weight_full_uncertainty():
    m = {}
    m[frozenset('B')] = tensor(0., device=DEVICE, dtype=DTYPE)
    m[frozenset('R')] = tensor(0., device=DEVICE, dtype=DTYPE)
    m[frozenset({'B','R'})] = tensor(1., device=DEVICE, dtype=DTYPE) # Uncertainty
    return m

def start_weights(s_list):
    list_initial_weights = []
    for s in s_list:
        m = {}
        m[frozenset('B')] = tensor(0.04, device=DEVICE, dtype=DTYPE, requires_grad=True)
        m[frozenset('R')] = tensor(0.06, device=DEVICE, dtype=DTYPE, requires_grad=True)
        m[frozenset({'B','R'})] = tensor(0.9, device=DEVICE, dtype=DTYPE, requires_grad=True) # Uncertainty
        optimizer = optim.Adam([m[frozenset('B')], m[frozenset('R')], m[frozenset({'B','R'})]])
        list_initial_weights.append([m, optimizer, s])

    return list_initial_weights


def read_rules(rule_set):
    s = "Rule {}: B = {}, R = {}, Uncertainty = {}"
    for i in range(len(rule_set)):
        dict_m = rule_set[i][0]
        b   = dict_m[frozenset({'B'})].item()
        r   = dict_m[frozenset({'R'})].item()
        r_b = dict_m[frozenset({'B', 'R'})].item()
        print(s.format(i+1,b,r,r_b))


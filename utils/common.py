
"""
@author: Tiago Roxo, UBI
@date: 2020
"""

from itertools import chain, combinations
from torch import tensor, optim
from utils.config import *


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


def order_rules_by_malign(rule_set):
    dict_rule_malig = {}
    for i in range(len(rule_set)):
        dict_rules = rule_set[i][0]
        dict_rule_malig[i+1] = dict_rules[frozenset({'M'})].item()
    dict_rule_malig_sorted = dict(sorted(dict_rule_malig.items(), key=lambda item: item[1], reverse=True))
    return dict_rule_malig_sorted


# ------------ Common -------------------

def get_class_plausibility(dict_m, dataset_name):
    if dataset_name == "A1_Dataset":
        class_0, class_1 = dict_m[frozenset({'B'})], dict_m[frozenset({'R'})]
    elif dataset_name == "BC_Dataset":
        class_0, class_1 = dict_m[frozenset({'B'})], dict_m[frozenset({'M'})]
    else:
        assert False
    return class_0, class_1 

def weight_full_uncertainty(dataset_name):
    m = {}
    if dataset_name == "A1_Dataset":
        m[frozenset('B')] = tensor(0., device=DEVICE, dtype=DTYPE)
        m[frozenset('R')] = tensor(0., device=DEVICE, dtype=DTYPE)
        m[frozenset({'B','R'})] = tensor(1., device=DEVICE, dtype=DTYPE) # Uncertainty
    elif dataset_name == "BC_Dataset": 
        m[frozenset('B')] = tensor(0., device=DEVICE, dtype=DTYPE, requires_grad=True)
        m[frozenset('M')] = tensor(0., device=DEVICE, dtype=DTYPE, requires_grad=True)
        m[frozenset({'B','M'})] = tensor(1.0, device=DEVICE, dtype=DTYPE, requires_grad=True) # Uncertainty
    else:
        assert False
    return m

def get_powerset_dataset(dataset_name):
    if dataset_name == "A1_Dataset":
        return A1_POWERSET
    elif dataset_name == "BC_Dataset":
        return BC_POWERSET
    else:
        assert False

def get_complete_set_dataset(dataset_name):
    if dataset_name == "A1_Dataset":
        return A1_COMPLETE_SET
    elif dataset_name == "BC_Dataset":
        return BC_COMPLETE_SET
    else:
        assert False

def start_weights(s_list, dataset_name):
    list_initial_weights = []
    if dataset_name == "A1_Dataset":
        for s in s_list:
            m = {}
            m[frozenset('B')] = tensor(0.04, device=DEVICE, dtype=DTYPE, requires_grad=True)
            m[frozenset('R')] = tensor(0.06, device=DEVICE, dtype=DTYPE, requires_grad=True)
            m[frozenset({'B','R'})] = tensor(0.9, device=DEVICE, dtype=DTYPE, requires_grad=True) # Uncertainty
            optimizer = optim.Adam([m[frozenset('B')], m[frozenset('R')], m[frozenset({'B','R'})]])
            list_initial_weights.append([m, optimizer, s])

    elif dataset_name == "BC_Dataset":
        for s in s_list:
            m = {}
            m[frozenset('B')] = tensor(0.04, device=DEVICE, dtype=DTYPE, requires_grad=True)
            m[frozenset('M')] = tensor(0.06, device=DEVICE, dtype=DTYPE, requires_grad=True)
            m[frozenset({'B','M'})] = tensor(0.9, device=DEVICE, dtype=DTYPE, requires_grad=True) # Uncertainty
            optimizer = optim.Adam([m[frozenset('B')], m[frozenset('M')], m[frozenset({'B','M'})]])
            list_initial_weights.append([m, optimizer, s])

    else:
        assert False

    return list_initial_weights
    
# --------- Rules ----------------

def generate_rule(index, mean, std, *att):
    r1 = lambda *att: att[index] <= mean-std
    r2 = lambda *att: mean-std < att[index] and att[index] <= mean
    r3 = lambda *att: mean < att[index] and att[index] <= mean+std
    r4 = lambda *att: att[index] > mean+std
    return [r1, r2, r3, r4]
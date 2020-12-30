
"""
@author: Tiago Roxo, UBI
@date: 2020
"""
from itertools import chain, combinations
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
        class_0, class_1 = dict_m[frozenset({'R'})], dict_m[frozenset({'B'})]
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


def get_powerset(set_elements):
    # Powerset: set + empty set + subsets of given set
    list_elements = list(set_elements)
    list_powerset = list(chain.from_iterable(combinations(list_elements, e) 
        for e in range(1, len(list_elements)+1))) # start at 1 to ignore empty set
    # Transform into a list of sets. 
    # We can use set() but then we will get "TypeError: unhashable type: 'set'" when adding as key to dictionary
    # So we use frozenset()
    list_sets_powerset = [frozenset(e) for e in list_powerset] # allow to be added to dictionary
    return list_sets_powerset

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
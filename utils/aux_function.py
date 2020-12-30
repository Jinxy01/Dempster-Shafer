
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


def order_rules_by_malign(rule_set):
    dict_rule_malig = {}
    for i in range(len(rule_set)):
        dict_rules = rule_set[i][0]
        dict_rule_malig[i+1] = dict_rules[frozenset({'M'})].item()
    dict_rule_malig_sorted = dict(sorted(dict_rule_malig.items(), key=lambda item: item[1], reverse=True))
    return dict_rule_malig_sorted





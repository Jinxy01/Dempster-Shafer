
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

def order_rules_by_malign(rule_set, dataset_name, malign):
    if dataset_name == "BC_Dataset":
        if malign:
            factor = frozenset({'M'})
        else:
            factor = frozenset({'B'})
    elif dataset_name == "HD_Dataset":
        if malign:
            factor = frozenset({'P'})
        else:
            factor = frozenset({'A'})
    else:
        assert False

    dict_rule_malig = {}
    for i in range(len(rule_set)):
        dict_rules = rule_set[i][0]
        dict_rule_malig[i+1] = dict_rules[factor].item()

    dict_rule_malig_sorted = dict(sorted(dict_rule_malig.items(), key=lambda item: item[1], reverse=True))
    return dict_rule_malig_sorted

def get_class_plausibility(dict_m, dataset_name):
    if dataset_name == "A1_Dataset":
        class_0, class_1 = dict_m[frozenset({'B'})], dict_m[frozenset({'R'})]
    elif dataset_name == "BC_Dataset":
        class_0, class_1 = dict_m[frozenset({'B'})], dict_m[frozenset({'M'})]
    elif dataset_name == "IRIS_Dataset":
        class_0, class_1, class_2 = dict_m[frozenset({'S'})], dict_m[frozenset({'C'})], dict_m[frozenset({'V'})]
        return class_0, class_1, class_2
    elif dataset_name == "HD_Dataset":
        class_0, class_1 = dict_m[frozenset({'A'})], dict_m[frozenset({'P'})]
    elif dataset_name == "WINE_Dataset":
        class_0, class_1, class_2 = dict_m[frozenset({'A'})], dict_m[frozenset({'B'})], dict_m[frozenset({'C'})]
        return class_0, class_1, class_2
    elif dataset_name == "DIG_Dataset":
    #   class_0, class_1, class_2, class_3, class_4, class_5, class_6, class_7, class_8, class_9 = dict_m[frozenset({'0'})], dict_m[frozenset({'1'})], dict_m[frozenset({'2'})], dict_m[frozenset({'3'})], dict_m[frozenset({'4'})], dict_m[frozenset({'5'})], dict_m[frozenset({'6'})], dict_m[frozenset({'7'})], dict_m[frozenset({'8'})], dict_m[frozenset({'9'})]
    #   return class_0, class_1, class_2, class_3, class_4, class_5, class_6, class_7, class_8, class_9
        class_0, class_1 = dict_m[frozenset({'0'})], dict_m[frozenset({'1'})]
    else:
        assert False
    return class_0, class_1 

# def weight_full_uncertainty(dataset_name):
#     m = {}
#     if dataset_name == "A1_Dataset":
#         m[frozenset('B')] = tensor(0., device=DEVICE, dtype=DTYPE)
#         m[frozenset('R')] = tensor(0., device=DEVICE, dtype=DTYPE)
#         m[frozenset({'B','R'})] = tensor(1., device=DEVICE, dtype=DTYPE) # Uncertainty
#     elif dataset_name == "BC_Dataset": 
#         m[frozenset('B')] = tensor(0., device=DEVICE, dtype=DTYPE, requires_grad=True)
#         m[frozenset('M')] = tensor(0., device=DEVICE, dtype=DTYPE, requires_grad=True)
#         m[frozenset({'B','M'})] = tensor(1.0, device=DEVICE, dtype=DTYPE, requires_grad=True) # Uncertainty
#     else:
#         assert False
#     return m

def get_powerset_dataset(dataset_name):
    if dataset_name == "A1_Dataset":
        return A1_POWERSET
    elif dataset_name == "BC_Dataset":
        return BC_POWERSET
    elif dataset_name == "IRIS_Dataset":
        return IRIS_POWERSET
    elif dataset_name == "HD_Dataset":
        return HD_POWERSET
    elif dataset_name == "WINE_Dataset":
        return WINE_POWERSET
    elif dataset_name == "DIG_Dataset":
        return DIG_POWERSET
    else:
        assert False

def get_complete_set_dataset(dataset_name):
    if dataset_name == "A1_Dataset":
        return A1_COMPLETE_SET
    elif dataset_name == "BC_Dataset":
        return BC_COMPLETE_SET
    elif dataset_name == "IRIS_Dataset":
        return IRIS_COMPLETE_SET
    elif dataset_name == "HD_Dataset":
        return HD_COMPLETE_SET
    elif dataset_name == "WINE_Dataset":
        return WINE_COMPLETE_SET
    elif dataset_name == "DIG_Dataset":
        return DIG_COMPLETE_SET
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

    elif dataset_name == "IRIS_Dataset":
        for s in s_list:
            m = {}
            m[frozenset('S')] = tensor(0.03, device=DEVICE, dtype=DTYPE, requires_grad=True) # Setosa
            m[frozenset('C')] = tensor(0.03, device=DEVICE, dtype=DTYPE, requires_grad=True) # cersiColour
            m[frozenset('V')] = tensor(0.04, device=DEVICE, dtype=DTYPE, requires_grad=True) # Virginica
            m[frozenset({'S','C','V'})] = tensor(0.9, device=DEVICE, dtype=DTYPE, requires_grad=True) # Uncertainty
            optimizer = optim.Adam([m[frozenset('S')], m[frozenset('C')], m[frozenset('V')],
                m[frozenset({'S','C','V'})]])
            list_initial_weights.append([m, optimizer, s])

    elif dataset_name == "HD_Dataset":
        for s in s_list:
            m = {}
            m[frozenset('A')] = tensor(0.04, device=DEVICE, dtype=DTYPE, requires_grad=True) # Disease Absent
            m[frozenset('P')] = tensor(0.06, device=DEVICE, dtype=DTYPE, requires_grad=True) # Disease Present
            m[frozenset({'A','P'})] = tensor(0.9, device=DEVICE, dtype=DTYPE, requires_grad=True) # Uncertainty
            optimizer = optim.Adam([m[frozenset('A')], m[frozenset('P')], m[frozenset({'A','P'})]])
            list_initial_weights.append([m, optimizer, s])

    elif dataset_name == "WINE_Dataset":
        for s in s_list:
            m = {}
            m[frozenset('A')] = tensor(0.03, device=DEVICE, dtype=DTYPE, requires_grad=True) # Wine A
            m[frozenset('B')] = tensor(0.03, device=DEVICE, dtype=DTYPE, requires_grad=True) # Wine B
            m[frozenset('C')] = tensor(0.04, device=DEVICE, dtype=DTYPE, requires_grad=True) # Wine C
            m[frozenset({'A','B','C'})] = tensor(0.9, device=DEVICE, dtype=DTYPE, requires_grad=True) # Uncertainty
            optimizer = optim.Adam([m[frozenset('A')], m[frozenset('B')], m[frozenset('C')],
                m[frozenset({'A','B','C'})]])
            list_initial_weights.append([m, optimizer, s])
    elif dataset_name == "DIG_Dataset":
        for s in s_list:
            m = {}
            m[frozenset('0')] = tensor(0.04, device=DEVICE, dtype=DTYPE, requires_grad=True)
            m[frozenset('1')] = tensor(0.06, device=DEVICE, dtype=DTYPE, requires_grad=True)
            # m[frozenset('2')] = tensor(0.01, device=DEVICE, dtype=DTYPE, requires_grad=True) 
            # m[frozenset('3')] = tensor(0.01, device=DEVICE, dtype=DTYPE, requires_grad=True) 
            # m[frozenset('4')] = tensor(0.01, device=DEVICE, dtype=DTYPE, requires_grad=True)
            # m[frozenset('5')] = tensor(0.01, device=DEVICE, dtype=DTYPE, requires_grad=True) 
            # m[frozenset('6')] = tensor(0.01, device=DEVICE, dtype=DTYPE, requires_grad=True) 
            # m[frozenset('7')] = tensor(0.01, device=DEVICE, dtype=DTYPE, requires_grad=True)
            # m[frozenset('8')] = tensor(0.01, device=DEVICE, dtype=DTYPE, requires_grad=True) 
            # m[frozenset('9')] = tensor(0.01, device=DEVICE, dtype=DTYPE, requires_grad=True) 
            # m[frozenset({'0','1','2','3','4','5','6','7','8','9'})] = tensor(0.9, device=DEVICE, dtype=DTYPE, requires_grad=True) # Uncertainty
            m[frozenset({'0','1'})] = tensor(0.9, device=DEVICE, dtype=DTYPE, requires_grad=True) # Uncertainty
            optimizer = optim.Adam([m[frozenset('0')], m[frozenset('1')], m[frozenset({'0','1'})]])
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

# def generate_rule_digits(index, mean, std, *att):
#     r1 = lambda *att: att[index] < mean-std or mean+std < att[index] 
#     r2 = lambda *att:  mean-std <= att[index] and att[index] <= mean+std
#     return [r1, r2]

def presentation_rule_helper(element, mean, std):
    return [
        RULE_LTE.format(element, mean-std),
        RULE_BETWEEN.format(mean-std, element, mean),
        RULE_BETWEEN.format(mean, element, mean+std),
        RULE_GT.format(element, mean+std)
    ]

# def presentation_rule_helper_digits(element, mean, std):
#     return [
#         RULE_INNER.format(mean-std, element, mean+std),
#         RULE_OUTER.format(element, mean-std, mean+std, element)
#     ]
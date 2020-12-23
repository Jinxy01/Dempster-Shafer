from itertools import chain, combinations
import torch
from utils.config import *

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

def dempster_rule(dict_m1, dict_m2):
    # Combine masses
    dict_combined_m = {}

    for s in POWERSET:
        sum_m = 0
        for s1 in dict_m1:
            for s2 in dict_m2:
                if s1.intersection(s2) == s and s1.intersection(s2) != EMPTY_SET:
                    sum_m += dict_m1[s1]*dict_m2[s2]
        dict_combined_m[s] = sum_m
    
    # Need to normalize so that sum = 1
    return normalize_masses_combined(dict_combined_m)

# ----------------------------------------------

def weight_full_uncertainty():
    m = {}
    m[frozenset('B')] = torch.tensor(0., device=DEVICE, dtype=DTYPE)
    m[frozenset('R')] = torch.tensor(0., device=DEVICE, dtype=DTYPE)
    m[frozenset({'B','R'})] = torch.tensor(1., device=DEVICE, dtype=DTYPE) # Uncertainty
    return m

def start_weights_dict(s_list):
    list_initial_weights = []
    for s in s_list:
        m = {}
        m[frozenset('B')] = torch.tensor(0.04, device=DEVICE, dtype=DTYPE, requires_grad=True)
        m[frozenset('R')] = torch.tensor(0.06, device=DEVICE, dtype=DTYPE, requires_grad=True)
        m[frozenset({'B','R'})] = torch.tensor(0.9, device=DEVICE, dtype=DTYPE, requires_grad=True) # Uncertainty
        list_initial_weights.append([m, s])

    return list_initial_weights
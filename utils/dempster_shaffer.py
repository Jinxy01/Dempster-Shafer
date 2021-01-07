
import torch
from utils.common import *

import numpy as np

def normalize_masses_combined(dict_combined_m):
    sum_m = 0
    for _, m in dict_combined_m.items():
        sum_m += m
    
    # It is already normalized or to avoid zero division
    if sum_m == 1.0 or sum_m == 0.0:
        return dict_combined_m
        
    dict_combined_m_norm = {}
    for s in dict_combined_m:
        dict_combined_m_norm[s] = dict_combined_m[s]/sum_m
    
    return dict_combined_m_norm

def dempster_rule(dict_m1, dict_m2, dataset_name):
    # Combine masses
    dict_combined_m = {}
    powerset = get_powerset_dataset(dataset_name)

    for s in powerset:
        sum_m = 0
        for s1 in dict_m1:
            for s2 in dict_m2:
                if s1.intersection(s2) == s and s1.intersection(s2) != EMPTY_SET:
                    sum_m += dict_m1[s1]*dict_m2[s2]
        dict_combined_m[s] = sum_m
    
    # Need to normalize so that sum = 1
    return normalize_masses_combined(dict_combined_m)


def belief_set(A, dict_m, dataset_name):
    sum_m = 0
    powerset = get_powerset_dataset(dataset_name)

    for s in powerset:
        if s.issubset(A):
            sum_m += dict_m[s]
    return sum_m


def belief(dict_m, dataset_name):
    dict_beliefs = {}
    complete_set = get_complete_set_dataset(dataset_name)

    for s in dict_m:
        if s == complete_set:
            continue
        dict_beliefs[s] = belief_set(s, dict_m)
    
    return dict_beliefs

# ---------------------------------

def plausibility_set(A, dict_m, dataset_name):
    sum_m = 0
    powerset = get_powerset_dataset(dataset_name)

    for s in powerset:
        if s.intersection(A) != EMPTY_SET:
            sum_m += dict_m[s]
    return sum_m

def plausibility(dict_m, dataset_name):
    dict_plausibility = {}
    complete_set = get_complete_set_dataset(dataset_name)

    for s in dict_m:
        if s == complete_set:
            continue
        dict_plausibility[s] = plausibility_set(s, dict_m, dataset_name)
    
    return dict_plausibility


def normalize_rule_set(rule_set):
    for i in range(len(rule_set)):
        dict_m = rule_set[i][0]
        rule_set[i][0] = normalize_masses_combined(dict_m)



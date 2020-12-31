
import torch
from utils.common import *

import numpy as np

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

def dempster_rule_working(dict_m1, dict_m2, dataset_name):
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

def commonality_2_classes(dict_m1, dict_m2, powerset):

    q_a = {}
    q_b = {}
    tot = len(powerset)
    for i in range(tot):
        if i == tot-1:
            q_a[powerset[i]] = dict_m1[powerset[i]]
            q_b[powerset[i]] = dict_m2[powerset[i]]
        else:
            q_a[powerset[i]] = dict_m1[powerset[i]] + dict_m1[powerset[tot-1]]
            q_b[powerset[i]] = dict_m2[powerset[i]] + dict_m2[powerset[tot-1]]

    return q_a, q_b


def dempster_rule(dict_m1, dict_m2, dataset_name):
    # Combine masses
    dict_combined_m = {}
    powerset = get_powerset_dataset(dataset_name)

    q_a, q_b = commonality_2_classes(dict_m1, dict_m2, powerset)
    tot = len(powerset)

    for s in powerset:
        dict_combined_m[s] = q_a[s]*q_b[s]
    
    for i in range(tot-1):
        dict_combined_m[powerset[i]] = dict_combined_m[powerset[i]] - dict_combined_m[powerset[tot-1]] 
    
    # We do not need to normalize now
    return dict_combined_m

# Their optimization...
# def dempster_rule_v2(dict_m1, dict_m2):
#     # Combine masses
#     r1, b1, r_b1 = dict_m1[frozenset({'R'})], dict_m1[frozenset({'B'})], dict_m1[frozenset({'R', 'B'})]
#     r2, b2, r_b2 = dict_m2[frozenset({'R'})], dict_m2[frozenset({'B'})], dict_m2[frozenset({'R', 'B'})] 
#     dict_combined_m = {}

#     dict_combined_m[frozenset({'R', 'B'})] = r_b1*r_b2
#     dict_combined_m[frozenset({'R'})]      = r1*r2 - dict_combined_m[frozenset({'R', 'B'})]
#     dict_combined_m[frozenset({'B'})]      = b1*b2 - dict_combined_m[frozenset({'R', 'B'})]
    
#     # Need to normalize so that sum = 1
#     return normalize_masses_combined(dict_combined_m)



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

def plausibility_working(dict_m, dataset_name):
    dict_plausibility = {}
    complete_set = get_complete_set_dataset(dataset_name)

    for s in dict_m:
        if s == complete_set:
            continue
        dict_plausibility[s] = plausibility_set(s, dict_m, dataset_name)
    
    return dict_plausibility

# optimized
def plausibility(dict_m, dataset_name):
    dict_plausibility = {}
    powerset = get_powerset_dataset(dataset_name)

    tot = len(powerset)
    for i in range(tot-1):
        dict_plausibility[powerset[i]] = dict_m[powerset[i]] + dict_m[powerset[tot-1]] 

    return dict_plausibility


def project_masses(list_dict_m):
    for rule_mass in list_dict_m:
        sum_m = 0
        for _, v in rule_mass[0].items():
            sum_m += v
  
        for k, v in rule_mass[0].items():
            rule_mass[0][k] = v + (1-sum_m)/3
    

# def project_masses_v2(list_tensor):
#     sum_m = 0
#     for e in list_tensor:
#         sum_m += e.item()

#     for i in range(len(list_tensor)):
#         list_tensor[i] = list_tensor[i] + (1-sum_m)/3
    
#     return list_tensor
    
def normalize_rule_set(rule_set):
    for i in range(len(rule_set)):
        dict_m = rule_set[i][0]
        rule_set[i][0] = normalize_masses_combined(dict_m)

# Testing stuff
# def find_normal_plane():
#     # Q = np.array([,1,2])
#     # R = np.array([-4,2,2])
#     # S = np.array([-2,1,5])
#     Q = np.array([1,0,0])
#     R = np.array([0,1,0])
#     S = np.array([0,0,1])

#     p = np.array([0.3146406412124634, -0.3311198651790619, 0.33113864064216614])
#     #print((R-Q))
#     #print((S-Q))
#     n = np.cross((R-Q) ,(S-Q))
#     print(n)

#     v = p
#     dist = v[0]*n[0] + v[1]*n[1] + v[2]*n[2]
#     p_ = p - dist*n
#     print(p_)
  

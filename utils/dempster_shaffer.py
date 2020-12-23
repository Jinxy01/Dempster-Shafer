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


def belief_set(A, dict_m):
    sum_m = 0
    for s in POWERSET:
        if s.issubset(A):
            sum_m += dict_m[s]
    return sum_m


def belief(dict_m):
    dict_beliefs = {}
    for s in dict_m:
        if s == COMPLETE_SET:
            continue
        dict_beliefs[s] = belief_set(s, dict_m)
    
    return dict_beliefs


def project_masses(list_dict_m):
    for rule_mass in list_dict_m:
        sum_m = 0
        for _, v in rule_mass[0].items():
            sum_m += v
  
        for k, v in rule_mass[0].items():
            rule_mass[0][k] = v + (1-sum_m)/3
    
    
    
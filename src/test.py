"""
@author: Tiago Roxo, UBI
@date: 2020
"""

from itertools import chain, combinations

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

def disease_p_a():
    set1 = {"P"}
    set2 = {"A"}
    return set1.union(set2)

def test():
    set_elements  = disease_p_a()
    list_powerset = get_powerset(set_elements)
    return list_powerset

######### DANGER ZONE ######### 
def get_belief_set(A, list_powerset, dict_m):
    sum_m = 0
    for s in list_powerset:
        if s.issubset(A):
            sum_m += dict_m[s]
    return sum_m


def get_belief(dict_m, list_powerset):
    dict_beliefs = {}
    for s in dict_m:
        if s == COMPLETE_SET:
            continue
        dict_beliefs[s] = get_belief_set(s, list_powerset, dict_m)
    
    return dict_beliefs

#  ------------

def get_plausibility_set(A, list_powerset, dict_m):
    sum_m = 0
    empty_set = set()
    for s in list_powerset:
        if s.intersection(A) != empty_set:
            sum_m += dict_m[s]
    return sum_m


def get_plausibility(dict_m, list_powerset):
    dict_plausibility = {}
    for s in dict_m:
        if s == COMPLETE_SET:
            continue
        dict_plausibility[s] = get_plausibility_set(s, list_powerset, dict_m)
    
    return dict_plausibility


########################### 

if __name__ == "__main__":
    list_powerset = test()
    m = {}
    COMPLETE_SET = frozenset({'A','P'})
    m[frozenset('P')] = 0.5
    m[frozenset('A')] = 0.2
    m[frozenset({'A','P'})] = 0.3
    #print(m)

    dict_beliefs      = get_belief(m, list_powerset)
    dict_plausibility = get_plausibility(m, list_powerset)
    print(dict_beliefs)
    print(dict_plausibility)
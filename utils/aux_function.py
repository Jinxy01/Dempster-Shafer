
from torch import tensor, optim
from utils.config import *
from utils.dempster_shaffer import *

def y_argmax(dict_m):
    return max(dict_m, key=(lambda key: dict_m[key]))

def y_argmax_train(dict_m):
    # Return probability for each classl using Cobb and Shenoy approach
    list_dict_m = list(dict_m.values())
    e1, e2 = list_dict_m
    list_dict_m[0] = e1/(e1+e2)
    list_dict_m[1] = e2/(e1+e2)
    return list_dict_m

def y_argmax_train_v2(dict_m):
    r, b, r_b = dict_m[frozenset({'R'})], dict_m[frozenset({'B'})], dict_m[frozenset({'R', 'B'})] 
    #max_m = max(r, b)
    #return r/(r + r_b), b/(b+r_b)
    return r, b, r_b




# Translates frozenset to class
def frozenset_to_class(y_hat):
    # For a1 and a2 dataset
    if y_hat == frozenset({'R'}):
        return 1 # Red is class 1
    return 0

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
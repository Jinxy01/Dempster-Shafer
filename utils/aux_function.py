
from torch import tensor, optim
from utils.config import *

def y_argmax(dict_m):
    return max(dict_m, key=(lambda key: dict_m[key]))

def y_argmax_train(dict_m):
    # Return predict for each class instead of only a number
    return list(dict_m.values())


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
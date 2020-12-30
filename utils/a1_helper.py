
from torch import tensor, optim
from utils.config import *

def read_rules_A1(rule_set):
    s = "Rule {}: B = {}, R = {}, Uncertainty = {}"
    for i in range(len(rule_set)):
        dict_m = rule_set[i][0]
        b   = dict_m[frozenset({'B'})].item()
        r   = dict_m[frozenset({'R'})].item()
        r_b = dict_m[frozenset({'B', 'R'})].item()
        print(s.format(i+1,b,r,r_b))

# def weight_full_uncertainty_A1():
#     m = {}
#     m[frozenset('B')] = tensor(0., device=DEVICE, dtype=DTYPE)
#     m[frozenset('R')] = tensor(0., device=DEVICE, dtype=DTYPE)
#     m[frozenset({'B','R'})] = tensor(1., device=DEVICE, dtype=DTYPE) # Uncertainty
#     return m

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

 # --------- Rules ----------------
 
def generate_rule_x(mean, std):
    r1 = lambda x,y: x <= mean-std
    r2 = lambda x,y: mean-std < x and x <= mean
    r3 = lambda x,y: mean < x and x <= mean+std
    r4 = lambda x,y: x > mean+std
    return [r1, r2, r3, r4]

def generate_rule_y(mean, std):
    r1 = lambda x,y: y <= mean-std
    r2 = lambda x,y: mean-std < y and y <= mean
    r3 = lambda x,y: mean < y and y <= mean+std
    r4 = lambda x,y: y > mean+std
    return [r1, r2, r3, r4]

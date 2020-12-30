
from torch import tensor, optim
from utils.config import *

# def read_rules_BC(rule_set):
#     for i in range(len(rule_set)):
#         dict_m  = rule_set[i][0]
#         ct      = dict_m[frozenset({'A'})].item()
#         ucsize  = dict_m[frozenset({'B'})].item()
#         ucshape = dict_m[frozenset({'C'})].item()
#         ma      = dict_m[frozenset({'D'})].item()
#         secz    = dict_m[frozenset({'E'})].item()
#         bn      = dict_m[frozenset({'F'})].item()
#         bc      = dict_m[frozenset({'G'})].item()
#         nn      = dict_m[frozenset({'H'})].item()
#         m       = dict_m[frozenset({'I'})].item()
#         u       = dict_m[frozenset({'A','B','C','D','E','F','G','H','I'})].item()

#         print(BC_READ_RULES.format(i+1,ct,ucsize,ucshape,ma,secz,bn,bc,nn,m,u))


# def weight_full_uncertainty_bc():
#     m = {}
#     m[frozenset('A')] = tensor(0., device=DEVICE, dtype=DTYPE, requires_grad=True)
#     m[frozenset('B')] = tensor(0., device=DEVICE, dtype=DTYPE, requires_grad=True)
#     m[frozenset('C')] = tensor(0., device=DEVICE, dtype=DTYPE, requires_grad=True)
#     m[frozenset('D')] = tensor(0., device=DEVICE, dtype=DTYPE, requires_grad=True)
#     m[frozenset('E')] = tensor(0., device=DEVICE, dtype=DTYPE, requires_grad=True)
#     m[frozenset('F')] = tensor(0., device=DEVICE, dtype=DTYPE, requires_grad=True)
#     m[frozenset('G')] = tensor(0., device=DEVICE, dtype=DTYPE, requires_grad=True)
#     m[frozenset('H')] = tensor(0., device=DEVICE, dtype=DTYPE, requires_grad=True)
#     m[frozenset('I')] = tensor(0., device=DEVICE, dtype=DTYPE, requires_grad=True)
#     m[frozenset({'A','B','C','D','E','F','G','H','I'})] = tensor(1.0, device=DEVICE, dtype=DTYPE, requires_grad=True) # Uncertainty
#     return m

# def start_weights(s_list):
#     list_initial_weights = []
#     for s in s_list:
#         m = {}
#         m[frozenset('A')] = tensor(0.01, device=DEVICE, dtype=DTYPE, requires_grad=True)
#         m[frozenset('B')] = tensor(0.01, device=DEVICE, dtype=DTYPE, requires_grad=True)
#         m[frozenset('C')] = tensor(0.01, device=DEVICE, dtype=DTYPE, requires_grad=True)
#         m[frozenset('D')] = tensor(0.01, device=DEVICE, dtype=DTYPE, requires_grad=True)
#         m[frozenset('E')] = tensor(0.01, device=DEVICE, dtype=DTYPE, requires_grad=True)
#         m[frozenset('F')] = tensor(0.01, device=DEVICE, dtype=DTYPE, requires_grad=True)
#         m[frozenset('G')] = tensor(0.01, device=DEVICE, dtype=DTYPE, requires_grad=True)
#         m[frozenset('H')] = tensor(0.01, device=DEVICE, dtype=DTYPE, requires_grad=True)
#         m[frozenset('I')] = tensor(0.01, device=DEVICE, dtype=DTYPE, requires_grad=True)
#         m[frozenset({'A','B','C','D','E','F','G','H','I'})] = tensor(0.91, device=DEVICE, dtype=DTYPE, requires_grad=True) # Uncertainty
#         optimizer = optim.Adam([m[frozenset('A')], m[frozenset('B')], m[frozenset('C')], m[frozenset('D')],
#             m[frozenset('E')], m[frozenset('F')], m[frozenset('G')], m[frozenset('H')], m[frozenset('I')],
#             m[frozenset({'A','B','C','D','E','F','G','H','I'})]])
#         list_initial_weights.append([m, optimizer, s])

#     return list_initial_weights

def read_rules_BC(rule_set):
    s = "Rule {}: B = {}, M = {}, Uncertainty = {}"
    for i in range(len(rule_set)):
        dict_m  = rule_set[i][0]
        b   = dict_m[frozenset({'B'})].item()
        m   = dict_m[frozenset({'M'})].item()
        b_m = dict_m[frozenset({'B','M'})].item()

        print(s.format(i+1,b,m, b_m))


def weight_full_uncertainty_bc():
    m = {}
    m[frozenset('B')] = tensor(0., device=DEVICE, dtype=DTYPE, requires_grad=True)
    m[frozenset('M')] = tensor(0., device=DEVICE, dtype=DTYPE, requires_grad=True)
    m[frozenset({'B','M'})] = tensor(1.0, device=DEVICE, dtype=DTYPE, requires_grad=True) # Uncertainty
    return m

def start_weights(s_list):
    list_initial_weights = []
    for s in s_list:
        m = {}
        m[frozenset('B')] = tensor(0.04, device=DEVICE, dtype=DTYPE, requires_grad=True)
        m[frozenset('M')] = tensor(0.06, device=DEVICE, dtype=DTYPE, requires_grad=True)
        m[frozenset({'B','M'})] = tensor(0.9, device=DEVICE, dtype=DTYPE, requires_grad=True) # Uncertainty
        optimizer = optim.Adam([m[frozenset('B')], m[frozenset('M')], m[frozenset({'B','M'})]])
        list_initial_weights.append([m, optimizer, s])

    return list_initial_weights

# --------- Rules ----------------

def generate_rule_ct(mean, std):
    r1 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: ct <= mean-std
    r2 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: mean-std < ct and ct <= mean
    r3 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: mean < ct and ct <= mean+std
    r4 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: ct > mean+std
    return [r1, r2, r3, r4]

def generate_rule_ucsize(mean, std):
    r1 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: ucsize <= mean-std
    r2 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: mean-std < ucsize and ucsize <= mean
    r3 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: mean < ucsize and ucsize <= mean+std
    r4 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: ucsize > mean+std
    return [r1, r2, r3, r4]

def generate_rule_ucshape(mean, std):
    r1 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: ucshape <= mean-std
    r2 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: mean-std < ucshape and ucshape <= mean
    r3 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: mean < ucshape and ucshape <= mean+std
    r4 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: ucshape > mean+std
    return [r1, r2, r3, r4]

def generate_rule_ma(mean, std):
    r1 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: ma <= mean-std
    r2 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: mean-std < ma and ma <= mean
    r3 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: mean < ma and ma <= mean+std
    r4 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: ma > mean+std
    return [r1, r2, r3, r4]

def generate_rule_secz(mean, std):
    r1 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: secz <= mean-std
    r2 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: mean-std < secz and secz <= mean
    r3 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: mean < secz and secz <= mean+std
    r4 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: secz > mean+std
    return [r1, r2, r3, r4]

def generate_rule_bn(mean, std):
    r1 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: bn <= mean-std
    r2 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: mean-std < bn and bn <= mean
    r3 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: mean < bn and bn <= mean+std
    r4 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: bn > mean+std
    return [r1, r2, r3, r4]

def generate_rule_bc(mean, std):
    r1 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: bc <= mean-std
    r2 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: mean-std < bc and bc <= mean
    r3 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: mean < bc and bc <= mean+std
    r4 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: bc > mean+std
    return [r1, r2, r3, r4]

def generate_rule_nn(mean, std):
    r1 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: nn <= mean-std
    r2 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: mean-std < nn and nn <= mean
    r3 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: mean < nn and nn <= mean+std
    r4 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: nn > mean+std
    return [r1, r2, r3, r4]

def generate_rule_m(mean, std):
    r1 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: m <= mean-std
    r2 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: mean-std < m and m <= mean
    r3 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: mean < m and m <= mean+std
    r4 = lambda ct,ucsize,ucshape,ma,secz,bn,bc,nn,m: m > mean+std
    return [r1, r2, r3, r4]
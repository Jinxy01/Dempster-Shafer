
from torch import tensor, optim

def read_rules_A1(rule_set):
    s = "Rule {}: B = {}, R = {}, Uncertainty = {}"
    for i in range(len(rule_set)):
        dict_m = rule_set[i][0]
        b   = dict_m[frozenset({'B'})].item()
        r   = dict_m[frozenset({'R'})].item()
        r_b = dict_m[frozenset({'B', 'R'})].item()
        print(s.format(i+1,b,r,r_b))


 # --------- Rules ----------------


# def generate_rule_x(mean, std):
#     r1 = lambda x,y: x <= mean-std
#     r2 = lambda x,y: mean-std < x and x <= mean
#     r3 = lambda x,y: mean < x and x <= mean+std
#     r4 = lambda x,y: x > mean+std
#     return [r1, r2, r3, r4]

# def generate_rule_y(mean, std):
#     r1 = lambda x,y: y <= mean-std
#     r2 = lambda x,y: mean-std < y and y <= mean
#     r3 = lambda x,y: mean < y and y <= mean+std
#     r4 = lambda x,y: y > mean+std
#     return [r1, r2, r3, r4]

from torch import tensor, optim

def read_rules_iris(rule_set):
    info = "Rule {}: S = {}, C = {}, V = {}, Uncertainty = {}"
    for i in range(len(rule_set)):
        dict_m = rule_set[i][0]
        s     = dict_m[frozenset({'S'})].item()
        c     = dict_m[frozenset({'C'})].item()
        v     = dict_m[frozenset({'V'})].item()
        s_c_v = dict_m[frozenset({'S', 'C', 'V'})].item()
        print(info.format(i+1,s,c,v,s_c_v))
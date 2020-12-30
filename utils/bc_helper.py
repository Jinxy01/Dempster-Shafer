
def read_rules_BC(rule_set):
    "scn", "ct", "ucsize", "ucshape", "ma", "secz", "bn", "bc", "nn", "m", "y"
    s = "Rule {}: scn = {}, ct = {}, ucsize = {}, ucshape = {}"
    for i in range(len(rule_set)):
        dict_m = rule_set[i][0]
        b   = dict_m[frozenset({'B'})].item()
        r   = dict_m[frozenset({'R'})].item()
        r_b = dict_m[frozenset({'B', 'R'})].item()
        print(s.format(i+1,b,r,r_b))



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
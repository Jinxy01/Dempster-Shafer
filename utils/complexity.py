
def ant_i(rule_id):
    condition_rule_1 = lambda x : ((x-1)/4).is_integer()
    condition_rule_4 = lambda x : ((x-4)/4).is_integer()
    return 2 if (condition_rule_1(rule_id) or condition_rule_4(rule_id)) else 1

def ant(rule_set):
    sum_ = 0
    for i in range(len(rule_set)):
        sum_ += ant_i(i+1)
    return sum_/(len(rule_set))

def q_ratr(rule_set, m):
    r = len(rule_set)
    sum_ = 0
    for i in range(r):
        sum_ += (ant_i(i+1)-1)/(m-1)
    return sum_/r

def q_atr(rule_set, m):
    _ant = ant(rule_set)
    return (_ant-1)/(m-1)

def q_fs(rule_set):
    return 0.06

def q_cplx(rule_set, m):
    q_ratr_ = q_ratr(rule_set, m)
    q_atr_  = q_atr(rule_set, m)
    q_fs_   = q_fs(rule_set)

    return (q_ratr_ + q_atr_ + q_fs_)/3
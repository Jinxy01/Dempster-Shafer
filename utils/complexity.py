
def ant_i(rule_id):
    # Based on defined rules
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

# ----------------------------
def associate_rule_to_attribute(dict_rules_att, num_att):
    dict_att = {x : 0 for x in range(num_att)}
    num_rule_per_att = 4
    for i in range(num_att):
        # Each attribute has 4 rules:
        for j in range(1, num_rule_per_att+1):
            dict_att[i] += dict_rules_att[i*num_rule_per_att+j]

    return dict_att

def get_num_rules_used(rule_set, dict_rules_att, *att):
    for i,(m,_,s) in enumerate(rule_set):
        if s(*att):
            dict_rules_att[i+1] += 1

def get_num_rules_per_att(rule_set, X_train, X_test, num_att):
    dict_rules_att = {x : 0 for x in range(1,len(rule_set)+1)}
    for att in X_train:
        get_num_rules_used(rule_set, dict_rules_att, *att)
    
    for att in X_test:
        get_num_rules_used(rule_set, dict_rules_att, *att)

    dict_att = associate_rule_to_attribute(dict_rules_att, num_att)
    return dict_att

def q_fs(rule_set, X_train, X_test, m):
    dict_att = get_num_rules_per_att(rule_set, X_train, X_test, m)
    n_fs = sum(dict_att.values())/(len(X_train)+len(X_test))
    sum_ = 0
    for i in range(m):
        sum_ = dict_att[i]
    
    return (n_fs-1)/(sum_-1)

# ------------------------------

def q_cplx(rule_set, X_train, X_test, m):
    q_ratr_ = q_ratr(rule_set, m)
    q_atr_  = q_atr(rule_set, m)
    q_fs_   = q_fs(rule_set, X_train, X_test, m)

    return (q_ratr_ + q_atr_ + q_fs_)/3
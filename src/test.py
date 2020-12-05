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

#  ------------------------

def get_plausibility_set(A, list_powerset, dict_m):
    sum_m = 0
    for s in list_powerset:
        if s.intersection(A) != EMPTY_SET:
            sum_m += dict_m[s]
    return sum_m


def get_plausibility(dict_m, list_powerset):
    dict_plausibility = {}
    for s in dict_m:
        if s == COMPLETE_SET:
            continue
        dict_plausibility[s] = get_plausibility_set(s, list_powerset, dict_m)
    
    return dict_plausibility

#  ------------------------

def normalize_masses_combined(dict_combined_m):
    sum_m = 0
    for _, m in dict_combined_m.items():
        sum_m += m
    
    # It is already normalized
    if sum_m == 1.0:
        return dict_combined_m

    dict_combined_m_norm = {}
    for s in dict_combined_m:
        dict_combined_m_norm[s] = dict_combined_m[s]/sum_m
    
    return dict_combined_m_norm

def combine_masses(dict_m1, dict_m2, list_powerset):
    dict_combined_m = {}

    for s in list_powerset:
        sum_m = 0
        for s1 in dict_m1:
            for s2 in dict_m2:
                if s1.intersection(s2) == s and s1.intersection(s2) != EMPTY_SET:
                    sum_m += dict_m1[s1]*dict_m2[s2]
        dict_combined_m[s] = sum_m
    
    # Need to normalize so that sum = 1
    print(dict_combined_m)
    return normalize_masses_combined(dict_combined_m)


# ---------------------------

def gradient_descent(theta_dg, learning_rate, X_train_complex, X_valid_complex):

    valid_error_list = []
    num_it = 0
    validation_error_prev = sys.maxsize
    max_interations_allowed = 2000

    mean_valid_error = 0
    sd_valid_error = 0
    v = []
    t = []

    while True:
        prediction = np.dot(X_train_complex,theta_dg)
        theta_dg = theta_dg -(1/len(Y_train)) * learning_rate * (X_train_complex.T.dot((prediction-Y_train)))

        validation_error = J(X_valid_complex,Y_valid, theta_dg)
        train_error = J(X_train_complex,Y_train, theta_dg)
        
        v.append(validation_error)
        t.append(train_error)

        valid_error_list.append(validation_error)
        # print(mean_valid_error, sd_valid_error)

        # Ref Engelbretch (IC), Eq 7.7, pag 96
        #if (validation_error > (mean_valid_error + sd_valid_error) and num_it > 1) or num_it >= max_interations_allowed:
        if validation_error > validation_error_prev or num_it >= max_interations_allowed:
        # if num_it >= max_interations_allowed:
            break

        mean_valid_error = np.mean(np.asarray(valid_error_list))
        sd_valid_error = np.std(np.asarray(valid_error_list))
        validation_error_prev = validation_error

        num_it += 1
    
    for i in range(len(t)):
      plt.plot(i, t[i], marker='o', color='black', linestyle='None')
      plt.plot(i, v[i], marker='x', color='red', linestyle='None')
        
    plt.show()

    return theta_dg, num_it


########################### 

if __name__ == "__main__":
    list_powerset = test()
    m1 = {}
    m2 = {}
    COMPLETE_SET = frozenset({'A','P'})
    EMPTY_SET    = set()
    # m1
    m1[frozenset('P')] = 0.5
    m1[frozenset('A')] = 0.2
    m1[frozenset({'A','P'})] = 0.3
    # m2
    m2[frozenset('P')] = 0
    m2[frozenset('A')] = 0.3
    m2[frozenset({'A','P'})] = 0.7

    dict_beliefs      = get_belief(m1, list_powerset)
    dict_plausibility = get_plausibility(m1, list_powerset)
    print(dict_beliefs)
    print(dict_plausibility)

    dict_combined_m = combine_masses(m1, m2, list_powerset)
    print(dict_combined_m)
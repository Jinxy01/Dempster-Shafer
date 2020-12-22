"""
@author: Tiago Roxo, UBI
@date: 2020
"""

from itertools import chain, combinations
from ref.adam import Adam as adam_optimizer
from main import aid_test
from dempster_shaffer import get_powerset
import numpy as np
import torch
import math
import sys

from dempster_shaffer import *

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

# --------------------------------------


def mse(y, y_hat):
    # Y_hat is the predicted one
    return (y - y_hat).pow(2).mean()

def is_converged(loss_current, loss_previous, tot_elements):
    epsilon=0.0001
    convergence = abs(loss_current-loss_previous) <= epsilon
    #print(np.size(convergence) - np.count_nonzero(convergence))
    # All rules have converged to minimal loss
    return np.count_nonzero(convergence) == tot_elements

def update_rule(R, B, R_B):
    m = {}
    m[frozenset('R')] = R
    m[frozenset('B')] = B
    m[frozenset({'R','B'})] = R_B
    return m

def get_rule(rules, y):
    if y <= -0.34:
        return 1
    elif y > -0.34 and y <= 0.04:
        return 2
    elif y > 0.04 and y <= 0.42:
        return 3
    else: # y > 0.42
        return 4

def get_rule_temp(rules, y):
    rule_id = -1
    if y <= 0:
        rule_id = 1
    else: # y > 0
        rule_id = 2

    return rules[rule_id][frozenset('B')], rules[rule_id][frozenset('R')], rules[rule_id][frozenset({'B','R'})]

def get_rule_temp_temp(rules, y):
    if y <= 0:
        return 1
    else: # y > 0
        return 2


def get_r_b_rb_rule(rule_dict):
    return rule_dict[frozenset('B')], rule_dict[frozenset('R')], rule_dict[frozenset({'B','R'})]


def project_masses(list_m):
    sum_m = 0
    for m in list_m:
        sum_m += m
    
    # It is already normalized
    if sum_m == 1.0:
        return list_m

    list_m_norm = []
    for m in list_m:
        list_m_norm.append(m/sum_m)
    
    return list_m_norm


def get_belief_confidence(r, b, r_b):
    dtype = torch.float
    device = torch.device("cpu")
    # Give confidence to prediction and account for uncertainty in such prediction
    # Red = 1 and Blue = -1
    if r > b:
        return r/(r+r_b)
    return -b/(b+r_b)


def y_agrmax_comb_m(belief_comb):
    return torch.tensor(1.) if belief_comb[frozenset('R')] > belief_comb[frozenset('B')] else torch.tensor(-1.)
    

def testing_stuff(X_train, Y_train, rules, list_powerset):

    dtype = torch.float
    device = torch.device("cpu")
    # device = torch.device("cuda:0")  # Uncomment this to run on GPU

    # Create Tensors to hold input and outputs.
    # By default, requires_grad=False, which indicates that we do not need to
    # compute gradients with respect to these Tensors during the backward pass.
    #x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
    #y = torch.sin(x)
    tot = len(X_train)

    # Create random Tensors for weights. For a third order polynomial, we need
    # 4 weights: y = a + b x + c x^2 + d x^3
    # Setting requires_grad=True indicates that we want to compute gradients with
    # respect to these Tensors during the backward pass.

    learning_rate = 1e-3
    previous_loss = sys.maxsize
    for t in range(10000):
        for i in range(tot):
            x, y = X_train[i]

            rule_id = get_rule_temp_temp(rules, y)
            dict_comb_masses = combine_masses(rules[rule_id], rules[3], list_powerset)
            r, b, r_b = get_r_b_rb_rule(dict_comb_masses)
            #print(r, b, r_b)
            # Forward pass: compute predicted y using operations on Tensors.
            #y_pred = a + b * x + c * x ** 2 + d * x ** 3

            #y_hat = get_belief_confidence(r, b, r_b)
            belief_comb = get_belief(dict_comb_masses, list_powerset)
            y_hat = y_agrmax_comb_m(belief_comb)

            #y_hat = (r + 0.5*r_b)/(r+b+r_b)
            #print(y_hat)
            
            # Compute and print loss using operations on Tensors.
            # Now loss is a Tensor of shape (1,)
            # loss.item() gets the scalar value held in the loss.
            # loss = (y_pred - y).pow(2).sum()
            y = Y_train[i] # Predicted class

            loss = mse(y, y_hat)
            
            if t % 100 == 99:
                print(t, loss.item())

            if is_converged(loss, previous_loss, 1):
                print("Breaking at {} iteration".format(t))
                break

            previous_loss = loss 
            
            loss.backward()
            r, b, r_b = get_r_b_rb_rule(rules[rule_id])
            with torch.no_grad():
                if r.grad is not None:
                    r -= learning_rate * r.grad
                if b.grad is not None:
                    b -= learning_rate * b.grad
                if r_b.grad is not None:
                    r_b -= learning_rate * r_b.grad

                # Manually zero the gradients after updating weights
                r.grad = None
                b.grad = None
                r_b.grad = None
            
            r, b, r_b = get_r_b_rb_rule(rules[3])
            with torch.no_grad():
                if r.grad is not None:
                    r -= learning_rate * r.grad
                if b.grad is not None:
                    b -= learning_rate * b.grad
                if r_b.grad is not None:
                    r_b -= learning_rate * r_b.grad

                # Manually zero the gradients after updating weights
                r.grad = None
                b.grad = None
                r_b.grad = None

    for _, weights in rules.items():
        r, b, r_b = weights[frozenset('B')], weights[frozenset('R')], weights[frozenset({'B','R'})]
        print(f'Result: r = {r.item()}, b = {b.item()}, r_b = {r_b.item()}')
        l_proj = project_masses([r.item(), b.item(), r_b.item()])
        print(l_proj)

def working_stuff(X_train, Y_train):

    dtype = torch.float
    device = torch.device("cpu")
    # device = torch.device("cuda:0")  # Uncomment this to run on GPU

    # Create Tensors to hold input and outputs.
    # By default, requires_grad=False, which indicates that we do not need to
    # compute gradients with respect to these Tensors during the backward pass.
    #x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
    #y = torch.sin(x)
    tot = len(X_train)

    # Create random Tensors for weights. For a third order polynomial, we need
    # 4 weights: y = a + b x + c x^2 + d x^3
    # Setting requires_grad=True indicates that we want to compute gradients with
    # respect to these Tensors during the backward pass.

    learning_rate = 1e-3
    previous_loss = sys.maxsize
    for t in range(2000):
        for i in range(tot):
            x, y = X_train[i]

            r, b, r_b = get_rule_temp(rules, y)
            #print(r, b, r_b)
            # Forward pass: compute predicted y using operations on Tensors.
            #y_pred = a + b * x + c * x ** 2 + d * x ** 3
            y_hat = get_belief_confidence(r, b, r_b)
            #y_hat = (r + 0.5*r_b)/(r+b+r_b)
            #print(y_hat)
            
            # Compute and print loss using operations on Tensors.
            # Now loss is a Tensor of shape (1,)
            # loss.item() gets the scalar value held in the loss.
            # loss = (y_pred - y).pow(2).sum()
            y = Y_train[i] # Predicted class

            loss = mse(y, y_hat)
            
            if t % 100 == 99:
                print(t, loss.item())

            if is_converged(loss, previous_loss, 1):
                print("Breaking at {} iteration".format(t))
                break

            previous_loss = loss 
            # Use autograd to compute the backward pass. This call will compute the
            # gradient of loss with respect to all Tensors with requires_grad=True.
            # After this call a.grad, b.grad. c.grad and d.grad will be Tensors holding
            # the gradient of the loss with respect to a, b, c, d respectively.
            loss.backward()

            # Manually update weights using gradient descent. Wrap in torch.no_grad()
            # because weights have requires_grad=True, but we don't need to track this
            # in autograd.
            with torch.no_grad():
                if r.grad is not None:
                    r -= learning_rate * r.grad
                if b.grad is not None:
                    b -= learning_rate * b.grad
                if r_b.grad is not None:
                    r_b -= learning_rate * r_b.grad

                # Manually zero the gradients after updating weights
                r.grad = None
                b.grad = None
                r_b.grad = None

    for _, weights in rules.items():
        r, b, r_b = weights[frozenset('B')], weights[frozenset('R')], weights[frozenset({'B','R'})]
        print(f'Result: r = {r.item()}, b = {b.item()}, r_b = {r_b.item()}')
        l_proj = project_masses([r.item(), b.item(), r_b.item()])
        print(l_proj)

#--------------------------

def get_initial_masses():
    dtype = torch.float
    device = torch.device("cpu")
    m = {}
    m[frozenset('B')] = torch.tensor(0.04, device=device, dtype=dtype, requires_grad=True)
    m[frozenset('R')] = torch.tensor(0.06, device=device, dtype=dtype, requires_grad=True)
    m[frozenset({'B','R'})] = torch.tensor(0.9, device=device, dtype=dtype, requires_grad=True) # Uncertainty
    return m

def start_rules():
    rules = {}
    rules[1] = get_initial_masses()
    rules[2] = get_initial_masses()
    rules[3] = get_initial_masses()
    #rules[4] = get_initial_masses()
    return rules

# --------------------------

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
    set1 = {"R"}
    set2 = {"B"}
    return set1.union(set2)

def test():
    set_elements  = disease_p_a()
    list_powerset = get_powerset(set_elements)
    return list_powerset

# ----------------------------------

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
    return normalize_masses_combined(dict_combined_m)

# ----------------------------------

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

# -------------------------------

if __name__ == "__main__":
    #X_train, Y_train, X_test, Y_test = aid_test()
    # Testing purposes
    EMPTY_SET    = set()
    COMPLETE_SET = frozenset({'R','B'})
    list_powerset = test()
    rules = start_rules()
    comb_masses = combine_masses(rules[1], rules[2], list_powerset)
    X_train = [np.array([-0.2, -0.3]), np.array([-0.3, 0.4])]
    Y_train = [-1, 1] 
    testing_stuff(X_train, Y_train, rules, list_powerset)
    exit(0)
    #print(rules)
    rules, current_loss_array, dict_it_rule = train(X_train, Y_train, rules)
    exit(0)
    print(rules)
    print(current_loss_array)
    print(dict_it_rule)
    #powerset = get_powerset({"B","R"})
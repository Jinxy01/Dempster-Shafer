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


def mse(Y, Y_hat):
    # Y_hat is the predicted one
    return np.square(np.subtract(Y,Y_hat)).mean() 

def is_converged(loss_current, loss_previous, tot_elements):
    epsilon=0.0001
    convergence = abs(loss_current-loss_previous) <= epsilon
    #print(np.size(convergence) - np.count_nonzero(convergence))
    # All rules have converged to minimal loss
    return np.count_nonzero(convergence) == tot_elements
    

def predict_y(m):
    y_hat = 1*m[frozenset('R')] + 0*m[frozenset('B')] + 0.5*m[frozenset({'R','B'})]
    return y_hat, [m[frozenset('R')], m[frozenset('B')], m[frozenset({'R','B'})]]

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

def project_masses(list_m):
    sum_m = 0
    for m in list_m:
        sum_m += m
    
    # It is already normalized
    if sum_m == 1.0:
        return list_m

    list_m_nomr = []
    for m in list_m:
        list_m_nomr.append(m/sum_m)
    
    return list_m_nomr


# Not working... need gradient of loss function...
def train_prev(X_train, Y_train, rules):
    converged = False
    adam = adam_optimizer(alpha=0.002)
    tot_elements = len(X_train)
    previous_loss = np.zeros(tot_elements)
    dict_it_rule = {1: 0, 2: 0, 3: 0, 4: 0} 
    it = 0
    while not converged:
        current_loss = []
        for i in range(tot_elements):
            [x, y] = X_train[i]
            #print(x, y)
            id_rule = get_rule(rules, y)
            dict_it_rule[id_rule] += 1
            # Not using MAF, yet
            m = rules[id_rule]
            y_hat, theta = predict_y(m)
            #print(Y_train[i])
            loss = mse(y_hat, Y_train[i])
            print(x, y, Y_train[i], loss)
            current_loss.append(loss)
            #print(theta)
            theta = adam.update(theta, loss, dict_it_rule[id_rule]) # Adam update basen on number of changes of the used rule
            #print(theta)
            #print(rules[id_rule])
            theta_projected = project_masses(theta)
            rules[id_rule] = update_rule(theta[0], theta[1], theta[2])

        current_loss_array = np.array(current_loss)
        converged = is_converged(current_loss, previous_loss, tot_elements)
        previous_loss = np.copy(current_loss_array)
        it += 1
        if it > 10:
            break
    return rules, current_loss_array, dict_it_rule

def train(X_train, Y_train, rules):
    converged = False
    adam = adam_optimizer(alpha=0.002)
    tot_elements = len(X_train)
    previous_loss = np.zeros(tot_elements)
    dict_it_rule = {1: 0, 2: 0, 3: 0, 4: 0} 
    it = 0

    dtype = torch.float
    device = torch.device("cpu")
    a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(optimizer)
    while e in range(10):
        current_loss = []
        for i in range(tot_elements):
            [x, y] = X_train[i]
            #print(x, y)
            id_rule = get_rule(rules, y)
            dict_it_rule[id_rule] += 1
            # Not using MAF, yet
            m = rules[id_rule]
            y_hat, theta = predict_y(m)
            #print(Y_train[i])
            loss = mse(y_hat, Y_train[i])
            print(x, y, Y_train[i], loss)
            current_loss.append(loss)
            #print(theta)
            theta = adam.update(theta, loss, dict_it_rule[id_rule]) # Adam update basen on number of changes of the used rule
            #print(theta)
            #print(rules[id_rule])
            theta_projected = project_masses(theta)
            rules[id_rule] = update_rule(theta[0], theta[1], theta[2])

        current_loss_array = np.array(current_loss)
        converged = is_converged(current_loss, previous_loss, tot_elements)
        previous_loss = np.copy(current_loss_array)
        it += 1
        if it > 10:
            break
    return rules, current_loss_array, dict_it_rule

def get_belief_t(r, b, r_b):
    dtype = torch.float
    device = torch.device("cpu")
    belief = max(r, b, r_b)
    #if r > b:
    #    return torch.tensor(1.0, device=device, dtype=dtype, requires_grad=True)
    #return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
    return belief

def get_belief_confidence(r, b, r_b):
    dtype = torch.float
    device = torch.device("cpu")
    # Give confidence to prediction and account for uncertainty in such prediction
    # Red = 1 and Blue = -1
    if r > b:
        return r/(r+r_b)
    return -b/(b+r_b)




def testing_stuff(X_train, Y_train):

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
    r   = torch.tensor(0.04, device=device, dtype=dtype, requires_grad=True)
    b   = torch.tensor(0.06, device=device, dtype=dtype, requires_grad=True)
    r_b = torch.tensor(0.9, device=device, dtype=dtype, requires_grad=True)

    learning_rate = 1e-3
    for t in range(2000):
        # Forward pass: compute predicted y using operations on Tensors.
        #y_pred = a + b * x + c * x ** 2 + d * x ** 3
        y_hat = get_belief_confidence(r, b, r_b)
        #y_hat = (r + 0.5*r_b)/(r+b+r_b)
        #print(y_hat)
        
        # Compute and print loss using operations on Tensors.
        # Now loss is a Tensor of shape (1,)
        # loss.item() gets the scalar value held in the loss.
        # loss = (y_pred - y).pow(2).sum()
        y = -1
        loss = (y_hat - y).pow(2).sum()
        print(loss, y_hat)
        if t % 100 == 99:
            print(t, loss.item())

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

    print(f'Result: y = {r.item()} + {b.item()} + {r_b.item()}')
    l_proj = project_masses([r.item(), b.item(), r_b.item()])
    print(l_proj)

#--------------------------

def get_initial_masses():
    m = {}
    m[frozenset('B')] = 0.04 # Blue
    m[frozenset('R')] = 0.06 # Red
    m[frozenset({'B','R'})] = 0.9 # Uncertainty
    return m

def start_rules():
    rules = {}
    rules[1] = get_initial_masses()
    rules[2] = get_initial_masses()
    rules[3] = get_initial_masses()
    rules[4] = get_initial_masses()
    return rules

if __name__ == "__main__":
    #X_train, Y_train, X_test, Y_test = aid_test()
    # Testing purposes
    X_train = [np.array([-0.2, -0.3]), np.array([-0.3, -0.4])]
    Y_train = [0, 1] 
    testing_stuff(X_train, Y_train)
    exit(0)
    rules = start_rules()
    #print(rules)
    rules, current_loss_array, dict_it_rule = train(X_train, Y_train, rules)
    exit(0)
    print(rules)
    print(current_loss_array)
    print(dict_it_rule)
    #powerset = get_powerset({"B","R"})
"""
@author: Tiago Roxo, UBI
@date: 2020
"""

from itertools import chain, combinations
from ref.adam import Adam as adam_optimizer
from main import aid_test
from dempster_shaffer import get_powerset
import numpy as np

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

def train(X_train, Y_train, rules):
    converged = False
    adam = adam_optimizer(alpha=0.002)
    tot_elements = len(X_train)
    previous_loss = np.zeros(tot_elements)
    it = 1

    while not converged:
        current_loss = []
        for i in range(tot_elements):
            [x, y] = X_train[i]
            #print(x, y)
            id_rule = get_rule(rules, y)
            # Not using MAF, yet
            m = rules[id_rule]
            y_hat, theta = predict_y(m)
            #print(Y_train[i])
            loss = mse(y_hat, Y_train[i])
            current_loss.append(loss)
            #print(theta)
            theta = adam.update(theta, loss, it) # Adam after for maybe....
            #print(theta)
            #print(rules[id_rule])
            rules[id_rule] = update_rule(theta[0], theta[1], theta[2])
        current_loss_array = np.array(current_loss)
        converged = is_converged(current_loss, previous_loss, tot_elements)
        previous_loss = np.copy(current_loss_array)
        it += 1
        if it > 2:
            break
    return rules, current_loss_array


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
    X_train, Y_train, X_test, Y_test = aid_test()
    rules = start_rules()
    print(rules)
    rules, current_loss_array = train(X_train, Y_train, rules)
    print(rules)
    print(current_loss_array)
    #powerset = get_powerset({"B","R"})
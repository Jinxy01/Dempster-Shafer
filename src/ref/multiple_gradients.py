import tensorflow.compat.v1 as tf # Changed
import csv
import numpy as np
import matplotlib.pyplot as plt
from random import uniform
import sys

# Slide 11/23, aula 2
def J(X, y, theta):
    # np.squeeze = Remove single-dimensional entries from the shape of an array. (Documentation)
    # [[[7,8,9]]] => [7,8,9]
    # This happens when using tensorflow
    preds = np.squeeze(np.matmul(X, theta))
    temp = preds - np.squeeze(y)
    # Nao precisa de sum para os exemplos deste codigo
    return np.sqrt(np.sum(np.matmul(np.transpose(temp), temp)))

#########################################################################
#   Read Data
#########################################################################

with open('pizza.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    next(csv_reader) # to skip the header file
    X = []
    y = []
    for row in csv_reader:
        X.append([float(row[0]), 1]) # 1 is the bias, slide 11/23 de aula 2
        y.append(float(row[1]))

X = np.asarray(X)
y = np.asarray(y)


#########################################################################
#   Brute Force
#########################################################################

def brute_force(X, y):
    min_erro = sys.maxsize # INT_MAX
    min_valor_theta_1 = 0
    min_valor_theta_2 = 0
    for valor_theta_1 in np.arange(0, 0.5, 0.01): # para evitar typerror
        for valor_theta_2 in np.arange(750, 1000, 0.1):
            erro_atual = J(X, y, np.array([valor_theta_1, valor_theta_2]))
            if erro_atual < min_erro:
                min_valor_theta_1 = valor_theta_1
                min_valor_theta_2 = valor_theta_2
                min_erro = erro_atual
    return np.array([min_valor_theta_1, min_valor_theta_2])


theta_brute_force = brute_force(X, y)
print('Solution (brute-force): J={:.1f}, Theta=({:.2f}, {:.2f})'.format(J(X,y, theta_brute_force), theta_brute_force[0], theta_brute_force[1]))


#########################################################################
#   Closed-form method
#########################################################################

theta_direct = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.transpose(X)), y) # Slide 12/23 de aula 2

print('Solution (closed-form): J={:.1f}, Theta=({:.2f}, {:.2f})'.format(J(X,y, theta_direct), theta_direct[0], theta_direct[1]))

#########################################################################
#  Batch Gradient Descent method
#########################################################################

def batch_grad_desc_fixed_numb_iter(theta_gd, learning_rate, tot_iterations):
    for i in range(tot_iterations):
        t_0 = 0
        t_1 = 0

        for j in range(len(y)):
            # Slide 17/23 de aula 2, tentar por em numpy
            t_0 += (theta_gd[0] * X[j][0] + theta_gd[1] - y[j]) * X[j][0]
            t_1 += theta_gd[0] * X[j][0] + theta_gd[1] - y[j]
        t_0 /= len(y)
        t_1 /= len(y)

        # Slide 16/23 de aula 2, tentar por em numpy
        # Prova de t_0 e t_1 ser derivate(J(theta)) => Slide 15/23 de aula 2, tentar por em numpy
        theta_gd[0] = theta_gd[0] - learning_rate * t_0
        theta_gd[1] = theta_gd[1] - learning_rate * t_1

    return theta_gd


def batch_grad_desc(theta_gd, learning_rate, epsilon):
    num_iter = 0
    while True:
        num_iter += 1
        t_0 = 0
        t_1 = 0
        for j in range(len(y)):
            # Slide 17/23 de aula 2, tentar por em numpy
            t_0 += (theta_gd[0] * X[j][0] + theta_gd[1] - y[j]) * X[j][0]
            t_1 += theta_gd[0] * X[j][0] + theta_gd[1] - y[j]

        t_0 /= len(y)
        t_1 /= len(y)

        # Slide 16/23 de aula 2, tentar por em numpy
        # Prova de t_0 e t_1 ser derivate(J(theta)) => Slide 15/23 de aula 2, tentar por em numpy
        theta_gd_0_ant = theta_gd[0]
        theta_gd_1_ant = theta_gd[1]
        theta_gd[0] = theta_gd[0] - learning_rate * t_0
        theta_gd[1] = theta_gd[1] - learning_rate * t_1
        if abs(theta_gd[0]-theta_gd_0_ant) < epsilon and abs(theta_gd[1]-theta_gd_1_ant) < epsilon:
            break

    return theta_gd, num_iter


#########################################################################
#  Stochastic Gradient Descent method
#########################################################################

# Ref = https://towardsdatascience.com/batch-linear-regression-9b3e863c2635
# Ref = https://towardsdatascience.com/batch-mini-batch-stochastic-gradient-descent-7a62ecba642a - Prefered
def stochastic_grad_desc_fixed_number_iter(theta_gd, learning_rate, tot_iterations):
    for i in range(tot_iterations):
        for j in range(len(y)):
            t_0 = (theta_gd[0] * X[j][0] + theta_gd[1] - y[j]) * X[j][0]
            t_1 = theta_gd[0] * X[j][0] + theta_gd[1] - y[j]
            # Update value of theta at each example of training set (X[j][0], to ignore the bias)
            theta_gd[0] = theta_gd[0] - learning_rate * t_0
            theta_gd[1] = theta_gd[1] - learning_rate * t_1
        
    return theta_gd


def stochastic_grad_desc(theta_gd, learning_rate, tot_iterations):
    num_iter = 0
    while True:
        num_iter += 1
        theta_gd_0_ant = theta_gd[0]
        theta_gd_1_ant = theta_gd[1]
        for j in range(len(y)):
            t_0 = (theta_gd[0] * X[j][0] + theta_gd[1] - y[j]) * X[j][0]
            t_1 = theta_gd[0] * X[j][0] + theta_gd[1] - y[j]
            # Update value of theta at each example of training set (X[j][0], to ignore the bias)
            theta_gd[0] = theta_gd[0] - learning_rate * t_0
            theta_gd[1] = theta_gd[1] - learning_rate * t_1

        if abs(theta_gd[0]-theta_gd_0_ant) < epsilon and abs(theta_gd[1]-theta_gd_1_ant) < epsilon:
            break
        
    return theta_gd, num_iter


#########################################################################
#  Adaptative Gradient Descent method
#########################################################################

# Ref = https://wiki.tum.de/display/lfdv/Adaptive+Learning+Rate+Method - Momentum
def batch_grad_adaptative_learning_rate_momentum(theta_gd, learning_rate, epsilon, gamma):
    num_iter = 0

    v_iter_ant_0 = 0
    v_iter_ant_1 = 0

    while True:
        num_iter += 1
        t_0 = 0
        t_1 = 0
        for j in range(len(y)):
            # Slide 17/23 de aula 2, tentar por em numpy
            t_0 += (theta_gd[0] * X[j][0] + theta_gd[1] - y[j]) * X[j][0]
            t_1 += theta_gd[0] * X[j][0] + theta_gd[1] - y[j]

        t_0 /= len(y)
        t_1 /= len(y)

        v_0 = gamma * v_iter_ant_0 + learning_rate * t_0
        v_1 = gamma * v_iter_ant_1 + learning_rate * t_1

        # Update
        theta_gd_0_ant = theta_gd[0]
        theta_gd_1_ant = theta_gd[1]

        theta_gd[0] = theta_gd[0] - v_0
        theta_gd[1] = theta_gd[1] - v_1

        v_iter_ant_0 = v_0
        v_iter_ant_1 = v_1

        if abs(theta_gd[0]-theta_gd_0_ant) < epsilon and abs(theta_gd[1]-theta_gd_1_ant) < epsilon:
            break

    return theta_gd, num_iter


# Ref = https://wiki.tum.de/display/lfdv/Adaptive+Learning+Rate+Method - Momentum
def batch_grad_adaptative_learning_rate_nesterov(theta_gd, learning_rate, epsilon, gamma):
    num_iter = 0

    v_iter_ant_0 = 0
    v_iter_ant_1 = 0

    while True:
        num_iter += 1
        t_0 = 0
        t_1 = 0
        # Update nao é feito a cada iteracao, logo fazer update antes de ciclo
        theta_gd[0] = theta_gd[0] - gamma * v_iter_ant_0
        theta_gd[1] = theta_gd[1] - gamma * v_iter_ant_1

        for j in range(len(y)):
            # Slide 17/23 de aula 2, tentar por em numpy
            t_0 += (theta_gd[0] * X[j][0] + theta_gd[1] - y[j]) * X[j][0]
            t_1 += theta_gd[0] * X[j][0] + theta_gd[1]- y[j]

        t_0 /= len(y)
        t_1 /= len(y)

        v_0 = gamma * v_iter_ant_0 + learning_rate * t_0
        v_1 = gamma * v_iter_ant_1 + learning_rate * t_1

        # Update
        theta_gd_0_ant = theta_gd[0]
        theta_gd_1_ant = theta_gd[1]

        theta_gd[0] = theta_gd[0] - v_0
        theta_gd[1] = theta_gd[1] - v_1

        v_iter_ant_0 = v_0
        v_iter_ant_1 = v_1

        if abs(theta_gd[0]-theta_gd_0_ant) < epsilon and abs(theta_gd[1]-theta_gd_1_ant) < epsilon:
            break

    return theta_gd, num_iter

# Ref = https://ruder.io/optimizing-gradient-descent/
def batch_grad_adaptative_learning_rate_adagrad(theta_gd, learning_rate, epsilon, epsilon_adagrad):
    num_iter = 0

    while True:
        num_iter += 1
        theta_gd_0_ant = theta_gd[0]
        theta_gd_1_ant = theta_gd[1]
        t_0 = 0
        t_1 = 0

        for j in range(len(y)):
            # Ref = aula 2 de ML, Proenca
            t_0 = (theta_gd[0] * X[j][0] + theta_gd[1] - y[j]) * X[j][0]
            t_1 = theta_gd[0] * X[j][0] + theta_gd[1] - y[j]

            # Ref = https://ruder.io/optimizing-gradient-descent/ - Adagrad
            gt = [t_0, t_1]
            G = np.zeros((len(gt), len(gt)), float)             
            np.fill_diagonal(G, np.sum(np.square(theta_gd))) # Neste caso, G é diagonal de 2x2, com ii = sum do quadrado de theta ( = [theta1, theta2])

            print("Den =",learning_rate/np.sqrt(G + epsilon_adagrad))
            theta_gd = theta_gd - np.dot(learning_rate/np.sqrt(G + epsilon_adagrad), gt)

        print(theta_gd)
        # Stopping condition
        if num_iter == 3:
            break
        if abs(theta_gd[0]-theta_gd_0_ant) < epsilon and abs(theta_gd[1]-theta_gd_1_ant) < epsilon:
            break
        
    return theta_gd, num_iter


# Get random values for theta
theta_gd = np.array([uniform(0., 0.5), uniform(750., 1000.)])
theta_gd_fixed_numb_iter = np.copy(theta_gd) # Copy to not get it changed inside functions
stochastic_theta = np.copy(theta_gd) # Copy to not get it changed inside functions
stochastic_theta_fixed_numb_iter = np.copy(theta_gd) # Copy to not get it changed inside functions
theta_momentum = np.copy(theta_gd) # Copy to not get it changed inside functions
theta_nesterov = np.copy(theta_gd) # Copy to not get it changed inside functions
theta_adagrad = np.copy(theta_gd) # Copy to not get it changed inside functions


learning_rate = 0.0000000001
tot_iterations = 100
epsilon = 0.000001
epsilon_adagrad = 0.00000001
gamma = 0.95 # https://wiki.tum.de/display/lfdv/Adaptive+Learning+Rate+Method - Momentum, Nesterov


#############
#theta_adagrad = batch_grad_adaptative_learning_rate_adagrad(theta_adagrad, 0.01, epsilon, epsilon_adagrad)
#print(theta_adagrad)
#exit(0)


stochastic_theta_fixed_numb_iter = stochastic_grad_desc_fixed_number_iter(stochastic_theta_fixed_numb_iter, learning_rate, tot_iterations)
print('Solution (Stochastic Gradient Descent, numb_iter={}): J={:.1f}, Theta=({:.2f}, {:.2f})'.format(tot_iterations, J(X,y, stochastic_theta_fixed_numb_iter), stochastic_theta_fixed_numb_iter[0], stochastic_theta_fixed_numb_iter[1]))

stochastic_theta, numb_iter = stochastic_grad_desc(stochastic_theta, learning_rate, epsilon)
print('Solution (Stochastic Gradient Descent, epsilon={}): J={:.1f}, Theta=({:.2f}, {:.2f}), Numb iterations={}'.format(epsilon, J(X,y, stochastic_theta), stochastic_theta[0], stochastic_theta[1], numb_iter))

theta_gd_fixed_numb_iter = batch_grad_desc_fixed_numb_iter(theta_gd_fixed_numb_iter, learning_rate, tot_iterations)
print('Solution (Batch Gradient Descent, numb_iter={}): J={:.1f}, Theta=({:.2f}, {:.2f})'.format(tot_iterations, J(X,y, theta_gd_fixed_numb_iter), theta_gd_fixed_numb_iter[0], theta_gd_fixed_numb_iter[1]))

theta_gd, numb_iter = batch_grad_desc(theta_gd, learning_rate, epsilon)
print('Solution (Batch Gradient Descent, epsilon={}): J={:.1f}, Theta=({:.2f}, {:.2f}), Numb iterations={}'.format(epsilon, J(X,y, theta_gd), theta_gd[0], theta_gd[1], numb_iter))

theta_momentum, numb_iter = batch_grad_adaptative_learning_rate_momentum(theta_momentum, learning_rate, epsilon, gamma)
print('Solution (Momentum Gradient Descent, epsilon={}): J={:.1f}, Theta=({:.2f}, {:.2f}), Numb iterations={}'.format(epsilon, J(X,y, theta_momentum), theta_momentum[0], theta_momentum[1], numb_iter))

theta_nesterov, numb_iter = batch_grad_adaptative_learning_rate_nesterov(theta_nesterov, learning_rate, epsilon, gamma)
print('Solution (Nesterov Gradient Descent, epsilon={}): J={:.1f}, Theta=({:.2f}, {:.2f}), Numb iterations={}'.format(epsilon, J(X,y, theta_nesterov), theta_nesterov[0], theta_nesterov[1], numb_iter))


exit(0)

#########################################################################
#   Tensor flow
#########################################################################

sess = tf.Session()
tf.disable_eager_execution() # Added

# Graph Definition

x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
y_target = tf.placeholder(shape=[None], dtype=tf.float32)
weight_0 = tf.Variable(tf.random.uniform(shape=[1, 1], minval=0., maxval=0.5))
weight_1 = tf.Variable(tf.random.uniform(shape=[1, 1], minval=750., maxval=1000))
weights = tf.concat([weight_0, weight_1], 0)

# Define the Model
with tf.variable_scope('model_definition') as scope:
    model_output = tf.matmul(x_data, weights)
    scope.reuse_variables()


def loss_l2(predict, gt):
    predict = tf.squeeze(predict)
    #predict = tf.Print(predict,["predict: ", tf.shape(predict)])
    resid = predict - gt
    ret = tf.sqrt(tf.reduce_sum(tf.pow(resid, tf.constant(2.))))
    return ret

loss = loss_l2(model_output, y_target)
my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(loss)

# Graph execution

init = tf.global_variables_initializer()
sess.run(init)


for i in range(tot_iterations):
    sess.run(train_step, feed_dict={x_data: X, y_target: y})

theta_tf = sess.run(weights)
cur_loss = J(X,y, theta_tf)

print('Solution (Tensor flow): J={:.1f}, Theta=({:.2f}, {:.2f})'.format(cur_loss, theta_tf[0][0], theta_tf[1][0]))


#########################################################################
#   TResults Visualization
#########################################################################

plt.ion()
plt.figure(1)
plt.plot(X[:,0], y, 'o')
plt.plot([np.min(X[:,0]), np.max(X[:,0])],[theta_direct[0]*np.min(X[:,0])+theta_direct[1], theta_direct[0]*np.max(X[:,0])+theta_direct[1] ],'-r', label='Closed-form')
plt.plot([np.min(X[:,0]), np.max(X[:,0])],[theta_gd[0]*np.min(X[:,0])+theta_gd[1], theta_gd[0]*np.max(X[:,0])+theta_gd[1] ],'-g', label='Gradient Descent')
plt.plot([np.min(X[:,0]), np.max(X[:,0])],[theta_tf[0]*np.min(X[:,0])+theta_tf[1], theta_tf[0]*np.max(X[:,0])+theta_tf[1]] ,'-b', label='Tensor flow')
plt.legend(loc='upper right')
plt.grid()
plt.show()
plt.pause(0.1)

input('Close app?')


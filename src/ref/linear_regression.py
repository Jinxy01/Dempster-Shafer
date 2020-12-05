import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable tensor flow warning / os.environ allows the disabling of all debugging logs
import tensorflow.compat.v1 as tf
import csv
import numpy as np
import matplotlib.pyplot as plt
from random import uniform
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize
import sys
import math

#########################################################################
#   Cost Function
#########################################################################

def J(X, y, theta):
    preds = np.squeeze(np.matmul(X, theta))
    temp =  preds - np.squeeze(y)
    #return np.sqrt(np.sum(np.matmul(np.transpose(temp), temp))/X.shape[0])
    return np.sum(np.matmul(np.transpose(temp), temp))/X.shape[0]


#########################################################################
#   Read Data
#########################################################################

input_file = 'EcologicalFootPrint.csv'
input_file_adapted = 'EcologicalFootPrint_adapted.csv'

# columns = ['crop_land', 'grazing_land', 'forest_land', 'fishing_ground', 'built_up_land', 'carbon', 'total']
columns = ['crop_land', 'grazing_land', 'forest_land', 'fishing_ground', 'built_up_land', 'carbon']
num_weights = len(columns)

df = pd.read_csv(input_file, usecols=columns, na_values='NULL') # Ler apenas as colunas necessárias e considerar 'NULL' como na values (para usar abaixo)
#df.dropna(inplace = True) # Remover linhas que contês valores NULL

index_record_to_be_read = []
index = 4
while index < df.shape[0]:
    index_record_to_be_read.append(index)
    index += 8

for column in columns:
    df[column] = df[column].fillna(value=df[column].mean()) # Substituir na_values pela média da coluna
df[df.index.isin(index_record_to_be_read)].to_csv(input_file_adapted, index=False) # Escrever novo ficheiro adaptado ao problema
# df.head(184).to_csv(input_file_adapted, index=False) # Escrever novo ficheiro adaptado ao problema


with open(input_file_adapted) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    X = []
    Y = []
    next(csv_reader) # to skip the header file
    # crop_land | grazing_land | forest_land | fishing_ground | built_up_land | carbon(Y) | total
    for row in csv_reader:
        # X.append([row[0], row[1], row[2], row[3], row[4], row[6], 1.0])
        X.append([row[0], row[1], row[2], row[3], row[4], 1.0])
        Y.append(float("{0:.3f}".format(float(row[5])))) # Carbon

X = np.asarray(X).astype(float)
Y = np.asarray(Y).astype(float)

#########################################################################
#   Produce 3 subsets (train, validation and test)
#########################################################################

X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2) # Split entre treino e os restantes (80%/20%)
X_valid, X_test, Y_valid, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5) # Split entre validação e teste (50%/50%)


X_train = np.asarray(X_train)
Y_train = Y_train.reshape(-1,1) # Passar a coluna, simular 2D (Matrix). Necessário para a normalize()
X_valid = np.asarray(X_valid)
Y_valid = Y_valid.reshape(-1, 1)
X_test = np.asarray(X_test)
Y_test = Y_test.reshape(-1, 1)

#########################################################################
#   Normalization
#########################################################################

X_train = normalize(X_train)
Y_train = np.squeeze(normalize(Y_train)) # retirar [] extra
X_valid = normalize(X_valid)
Y_valid = np.squeeze(normalize(Y_valid))
X_test = normalize(X_test)
Y_test = np.squeeze(normalize(Y_test))

#########################################################################
#   Closed-form method
#########################################################################

theta_direct = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X_train), X_train)), np.transpose(X_train)), Y_train)

# print('Solution (Closed-Form): J={:.1f}, Theta=({:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(J(X_test,Y_test, theta_direct), np.asscalar(theta_direct[0]), np.asscalar(theta_direct[1]), np.asscalar(theta_direct[2]), np.asscalar(theta_direct[3]), np.asscalar(theta_direct[4]), np.asscalar(theta_direct[5]), np.asscalar(theta_direct[6])))
print('Solution (Closed-Form): J={:.1f}, Theta=({:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(J(X_test,Y_test, theta_direct), np.asscalar(theta_direct[0]), np.asscalar(theta_direct[1]), np.asscalar(theta_direct[2]), np.asscalar(theta_direct[3]), np.asscalar(theta_direct[4]), np.asscalar(theta_direct[5])))

#########################################################################
#   Gradient Descent method
#########################################################################


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


def add_complexity(complexity, X):
    X_list = []
    for i in range(2,complexity+1):
        X_complex_i = [[float(x)**i for x in row] for row in X]
        X_list.append(X_complex_i)
    
    for X_complex in X_list:
        X = np.concatenate([X, X_complex], axis=1)

    return X

def model_complexity(complexity, X_train, X_valid, X_test):
    X_train = add_complexity(complexity, np.squeeze(X_train).tolist())
    X_valid = add_complexity(complexity, np.squeeze(X_valid).tolist())
    X_test = add_complexity(complexity, np.squeeze(X_test).tolist())
    return np.array(X_train), np.array(X_valid), np.array(X_test)


def complexity_given_learning_rate(learning_rate, max_complexity):
    custo_list = []
    custo_direct_list = []

    for complexity in range(1,max_complexity):
        #print("***************************************************************")
        #print("Complexity=",complexity)

        X_train_complex, X_valid_complex, X_test_complex = model_complexity(complexity, X_train, X_valid, X_test)
        theta_dg = [uniform(np.amin(X_train_complex[:,i]), np.amax(X_train_complex[:,i])) for i in range(num_weights*complexity)]
        theta_dg, it = gradient_descent(theta_dg, learning_rate, X_train_complex, X_valid_complex)
        custo =  float("{0:.3f}".format(J(X_test_complex,Y_test, theta_dg)))

        #theta_direct = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X_train_complex), X_train_complex)), np.transpose(X_train_complex)), Y_train)
        #print('Solution (Closed-Form):      J={:.1f}'.format(J(X_test_complex,Y_test, theta_direct)))
        #custo_direct =  float("{0:.3f}".format(J(X_test_complex,Y_test, theta_direct)))

        custo_list.append(custo)
        #custo_direct_list.append(custo_direct)

        #print('Solution (Gradient Descent): J={:.1f}'.format(J(X_test_complex,Y_test, theta_dg)))

    return custo_list, custo_direct_list


learning_rate = 0.1
max_complexity = 5
num_it = 1

while True:
    custo_list = []
    complexity_list = [i for i in range(1,max_complexity)]
    custo_direct_list = []

    print("LR =",learning_rate)
    for j in range(num_it):
        if len(custo_list) == 0:
            custo_list, custo_direct_list = complexity_given_learning_rate(learning_rate, max_complexity)
        else:
            custo_list_temp, custo_direct_list_temp = complexity_given_learning_rate(learning_rate, max_complexity)
            for i in range(len(custo_list)):
                custo_list[i] += custo_list_temp[i]
                #custo_direct_list[i] += custo_direct_list_temp[i]
    
    # Media
    custo_list = [i/num_it for i in custo_list]
    #custo_direct_list = [i/num_it for i in custo_direct_list]


    fig, ax = plt.subplots()
    ax.scatter(complexity_list, custo_list)
    for i in range(len(custo_list)):
        if custo_list[i] == min(custo_list):
            ax.annotate(custo_list[i], (complexity_list[i], custo_list[i]))
            # plt.plot(row[0], row[1], marker='x', color='red')

    if learning_rate >= 0.1:
        learning_rate += 0.2
    else:
        learning_rate *= 10

    plt.grid()
    plt.show(block=False)
    plt.pause(0.1)
    break

input("Done!") 
exit(0)


#########################################################################
#   Tensor flow
#########################################################################

learning_rate = 0.0000001
tot_iterations = 100
sess = tf.Session()
tf.disable_eager_execution()

# Graph Definition

x_data = tf.placeholder(shape=[None, num_weights], dtype=tf.float32)
y_target = tf.placeholder(shape=[None], dtype=tf.float32)

weights_list = [tf.Variable(tf.random.uniform(shape=[1, 1], minval=np.amin(X_train[:,i]), maxval=np.amax(X_train[:,i]))) for i in range(num_weights)]
weights = tf.concat(weights_list, axis=0)

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
    sess.run(train_step, feed_dict={x_data: X_train, y_target: Y_train})

theta_tf = sess.run(weights)
cur_loss = J(X_test,Y_test, theta_tf)

print('Solution (Tensor flow): J={:.1f}, Theta=({:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(cur_loss, theta_tf[0][0], theta_tf[1][0], theta_tf[2][0], theta_tf[3][0], theta_tf[4][0], theta_tf[5][0], theta_tf[6][0]))


# #########################################################################
# #   TResults Visualization
# #########################################################################

# plt.ion()
# plt.figure(1)
# plt.plot(X[:,0], y, 'o')
# plt.plot([np.min(X[:,0]), np.max(X[:,0])],[theta_direct[0]*np.min(X[:,0])+theta_direct[1], theta_direct[0]*np.max(X[:,0])+theta_direct[1] ] ,'-r', label='Closed-form')
# plt.plot([np.min(X[:,0]), np.max(X[:,0])],[theta_gd[0]*np.min(X[:,0])+theta_gd[1], theta_gd[0]*np.max(X[:,0])+theta_gd[1] ] ,'-g', label='Gradient Descent')
# plt.plot([np.min(X[:,0]), np.max(X[:,0])],[theta_tf[0]*np.min(X[:,0])+theta_tf[1], theta_tf[0]*np.max(X[:,0])+theta_tf[1] ] ,'-b', label='Tensor flow')
# plt.legend(loc='upper right')
# plt.grid()
# plt.show()
# plt.pause(0.1)

# input('Close app?')

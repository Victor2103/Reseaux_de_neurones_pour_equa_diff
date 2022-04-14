import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math as ma


# Parameters
learning_rate = 0.01

# Network Parameters
n_input = 2  # input layer number of neurons
n_hidden_1 = 32 # 1st layer number of neurons
n_hidden_2 = 32 # 2nd layer number of neurons
n_hidden_3 = 32 # 3th layer number of neurons
n_hidden_4 = 32 # 4th layer number of neurons
n_hidden_5 = 32 # 5th layer number of neurons
n_output = 1    # output layer number of neurons

weights = {
    'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.random.normal([n_hidden_3, n_hidden_4])),
    'h5': tf.Variable(tf.random.normal([n_hidden_4, n_hidden_5])),
    'out': tf.Variable(tf.random.normal([n_hidden_5, n_output]))
}
biases = {
    'b1': tf.Variable(tf.random.normal([n_hidden_1])),
    'b2': tf.Variable(tf.random.normal([n_hidden_2])),
    'b3': tf.Variable(tf.random.normal([n_hidden_3])),
    'b4': tf.Variable(tf.random.normal([n_hidden_4])),
    'b5': tf.Variable(tf.random.normal([n_hidden_5])),
    'out': tf.Variable(tf.random.normal([n_output]))
}

# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.SGD(learning_rate)



# Create model
def multilayer_perceptron(x,t):
    x = np.array([[x,t]],  dtype='float32')
    # Hidden fully connected layer with 32 neurons
    layer_1 = tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    # Hidden fully connected layer with 32 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    # Hidden fully connected layer with 32 neurons
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)
    # Hidden fully connected layer with 32 neurons
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.sigmoid(layer_4)
    # Hidden fully connected layer with 32 neurons
    layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
    layer_5 = tf.nn.sigmoid(layer_5)
    # Output fully connected layer
    output = tf.matmul(layer_5, weights['out']) + biases['out']
    return (tf.nn.sigmoid(output))

# Universal Approximator
def g(x,t):
  return 1+multilayer_perceptron(x,t)



def loss_apprentissage():
    X=xcoord
    summation=[]
    for i in range(len(X)):
        summation.append((g(X[i],0)-f[i])**2)
    return tf.reduce_sum(tf.abs(summation))


def apprentissage():
    with tf.GradientTape() as tape:
        loss = loss_apprentissage()
    trainable_variables = list(weights.values()) + list(biases.values())
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return(loss)




u_x0 = 1
u_x2 = 1


def u(x):
    if (x>=0.5) and (x<=1):
        return 2
    else :
        return 1



def training_loss(exa):
    summation = []
    #T=0.09 #Time fixed
    X=xcoord
    Ttab=[0,0.1,0.2,0.3,0.4,0.5,0.6]
    fTab=[0,0.1,0.2,0.3,0.4,0.5]
    for j in range(len(fTab)):
        for i in range(0,X.shape[0]-1):
            #dNN_t = (g(X[i],T+dt)-g(X[i],T))/dt
            #dNN_x = (g(X[i+1],T) - g(X[i],T)) / (X[i+1] - X[i])
            #dNN_t = (g(X[i],T[j+1]) - g(X[i],T[j])) / (T[j+1] - T[j])
            #summation.append((dNN_x + dNN_t)**2 + (g(0,T) - u_x0)**2 + (g(2,T) - u_x2)**2 + (g(X[i],0) - u(X[i]))**2) #For Convection
            summation.append((g(X[i],fTab[j]) - exa[j,i])**2) 
    for i in Ttab:
        summation.append((g(0,i) - u_x0)**2 + (g(2,i) - u_x2)**2)
    return tf.reduce_sum(tf.abs(summation))

def train_step(exa):
    with tf.GradientTape() as tape:
        loss = training_loss(exa)
    trainable_variables = list(weights.values()) + list(biases.values())
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return(loss)




#Initialisation of the function at t=0
def Initialisation(nx):
    xcoord = np.ones(nx)
    tmp=0
    for i in range(nx):
        xcoord[i]=tmp
        tmp=tmp+dx
    f=np.ones(nx)
    for i in range(len(f)):
        f[i]=u(xcoord[i])
    return(xcoord,f)









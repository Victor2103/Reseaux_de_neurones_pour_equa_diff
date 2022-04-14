import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math as ma


# initial condition
u_x0 = 1
u_x2 = 1
dt=0.1
# infinitesimal small number
inf_s = np.sqrt(np.finfo(np.float32).eps)

# Parameters
learning_rate = 0.01
training_steps = 5000
batch_size = 1000
display_step = training_steps/20

# Network Parameters
n_input = 2  # input layer number of neurons
n_hidden_1 = 20 # 1st layer number of neurons
n_hidden_2 = 20 # 2nd layer number of neurons
n_hidden_3 = 20 # 3th layer number of neurons
n_hidden_4 = 20 # 4th layer number of neurons
n_hidden_5 = 20 # 5th layer number of neurons
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
    return (tf.nn.sigmoid(output)+1)

# Universal Approximator
def g(x,t):
  return multilayer_perceptron(x,t);

#Initial condition
def u(x):
    if (x>=0.5) and (x<=1):
        return 2
    else :
        return 1

    
# Custom loss function to approximate the derivatives
def custom_loss():
    summation = []
    T=0 #Time fixed
    X=np.array([0.49,0.5,0.51,0.75,0.99,1,1.01])
    for i in range(0,X.shape[0]-1):
            dNN_t = (g(X[i],T+dt)-g(X[i],T))/dt
            dNN_x = (g(X[i+1],T) - g(X[i],T)) / (X[i+1] - X[i])
            #dNN_t = (g(X[i],T[j+1]) - g(X[i],T[j])) / (T[j+1] - T[j])
            summation.append((dNN_x + dNN_t)**2 + (g(X[i],0) - u(X[i]))**2 + (g(0,T) - u_x0)**2 + (g(2,T) - u_x2)**2) #For Convection
    return tf.reduce_sum(tf.abs(summation))
    # return tf.sqrt(tf.reduce_mean(tf.abs(summation)))



def train_step():
    with tf.GradientTape() as tape:
        loss = custom_loss()
    trainable_variables = list(weights.values()) + list(biases.values())
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return(loss)



for i in range(training_steps):
    train_step()
    if i % display_step == 0:
        print("loss: %f " % (custom_loss()))



#Initialisation of the function at t=0
def Initialisation(nx):
    xcoord = np.ones(nx)
    tmp=0
    for i in range(nx):
        xcoord[i]=tmp
        tmp=tmp+dx
    f=np.ones(nx)
    for i in range(len(f)):
        if ((xcoord[i]>=0.5) and (xcoord[i]<=1)) :
            f[i]=2
    return(xcoord,f)

#Function for application of the finite difference method 
def Apply_method(f,nx):
    u1=f.copy()
    for i in range(1,nx):
        u1[i]=f[i] - dt/dx*(f[i]-f[i-1])
    return(u1)

def Apply_for_all(f,nx,nt):
    tmp=0
    for j in range(nt):
        plt.plot(xcoord,f,label="At t="+str(tmp))
        plt.legend(loc=2, prop={'size': 10})
        plt.show()
        plt.close()
        f=Apply_method(f,nx)
        tmp=tmp+dt
    return(f)

#X beetween 0 and 2 
nx= 50 #Twenty points for x
nt=10 #T is divided by 10
dx = 2/(nx-1)
dt = 0.1

xcoord,f = Initialisation(nx)
#f=Apply_for_all(f,nx,nt)

#Display of the last graph
plt.plot(xcoord,f,label="Final")
plt.legend(loc=2, prop={'size': 10})
plt.show()
plt.close()



from matplotlib.pyplot import figure

figure(figsize=(10,10))
# True Solution (found analitically)
def true_solution(x,t):
    return u(x-t)

X = np.linspace(0,2, 50)

T=0
result = []
for i in xcoord:
  #result.append(f(i))
  result.append(g(i,T).numpy()[0][0])

S=[]
for i in X:
    S.append(true_solution(i,T))

plt.plot(xcoord,f,label="At t=0")
  
#plt.plot(X, S, label="Original Function")
plt.plot(X, result, label="Neural Net Approximation")
plt.legend(loc=2, prop={'size': 20})
plt.show()
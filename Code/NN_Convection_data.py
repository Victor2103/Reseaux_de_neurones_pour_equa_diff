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



nx= 50
dx = 2/(nx-1)
dt2=0.001

xcoord,f = Initialisation(nx)



apprentissage_steps=2000
display_step = apprentissage_steps/20
for i in range(apprentissage_steps):
    apprentissage()
    if i % display_step == 0:
        print(loss_apprentissage())



from matplotlib.pyplot import figure

figure(figsize=(10,10))



result=[]
T=0
for i in xcoord:
  #result.append(f(i))
  result.append(g(i,T).numpy()[0][0])

plt.plot(xcoord,f,label="analytical solution at t=0")
plt.plot(xcoord, result, label="Neural Net Approximation at t=0")
plt.legend(loc=1, prop={'size': 10})
plt.show()
plt.close()



def Apply_method(f,nx):
    u1=f.copy()
    for i in range(1,nx):
        u1[i]=f[i] - dt2/dx*(f[i]-f[i-1])
    return(u1)
  
tabfinal=np.ones((6,nx))
i=1
tabfinal[0]=f
deb=0
obj=0.1
while (i<6):
    tabfinal[i]=tabfinal[i-1]
    while (deb<obj):
        tabfinal[i]=Apply_method(tabfinal[i],nx)
        deb=dt2+deb
    i=i+1
    obj=obj+0.1



training_steps = 1000
display_step = training_steps/5
#print("for T= "+str(T))
for i in range(training_steps):
    train_step(tabfinal)
    if i % display_step == 0:
        print("loss: %f " % (training_loss(tabfinal)))



X = xcoord
def true_solution(x,t):
    return u(x-t)

result=[]
T=0
T2=0.5
for i in xcoord:
  result.append(g(i,T2).numpy()[0][0])


S=[]
for i in X:
    S.append(true_solution(i,T))

S2=[]
for i in X:
    S2.append(true_solution(i,T2))
tableautemps=[0,0.1,0.2,0.3,0.4,0.5]
figure(figsize=(10,10))
for i in range(len(tableautemps)):
    result=[]
    for j in xcoord:
        result.append(g(j,tableautemps[i]).numpy()[0][0])
    plt.plot(X, result, label="Neural Net Approximation t="+str(tableautemps[i]))
    plt.plot(xcoord,tabfinal[i],label="Lorena's Barba Solution t="+str(tableautemps[i]))
    
plt.legend(loc=1, prop={'size': 10})    
plt.show()
plt.close()


f=Apply_method(tabfinal[5],nx)
T=0.501
while (T<0.55):
    f=Apply_method(f,nx)
    T=T+0.001


result=[]
T=0.5
T2=0.55

for i in xcoord:
  result.append(g(i,T2).numpy()[0][0])

figure(figsize=(10,10))
#plt.plot(X,S,label="true solution t="+str(T))
plt.plot(xcoord, result, label="Neural Net Approximation t="+str(T2))
plt.plot(xcoord,tabfinal[5],label="Lorena's Barba Solution t="+str(T))
plt.plot(xcoord,f,label="Lorena's Barba Solution t="+str(T2))
plt.legend(loc=2, prop={'size': 10})
plt.show()
plt.close()








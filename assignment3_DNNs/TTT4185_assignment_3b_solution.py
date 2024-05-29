# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # TTT4185 Machine learning for Speech technology
#
# ## Computer assignment 3b:  Regression analysis
#
# Regression analysis is used to estimate/measure the relationship between an _independent_ variable, say $x$, and a _dependent_ variable, say $y$. One of the simplest regression problems is 
# \begin{equation}
# y = ax + b
# \end{equation}
# where $a$ and $b$ are constants. In practice our observations will be contaminated by noise, so we have
# \begin{equation}
# y = ax + b + n,
# \end{equation}
# where $n$ is noise, eg. measurement errors. This particular problem is called _linear regression_.
#
# We will have a look at _non-linear regression_, using deep neural networks. Here we are looking at general regression problems in the form 
# \begin{equation}
# y = f(x) + n,
# \end{equation}
# and our goal will be to try and approximate the unknown function $f()$ based on observations $\{(x_i,y_i)\}$. We start by defining our "unknown" function $f$ :

# +
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


# +
# Define function
def f(x):
    return x**2 + np.cos(20*x)*np.sign(x)

# Setup some simulation parameters
# Number of observations
N = 5000

# Plot a "clean" version of the relationship between x and y
plt.figure(figsize=(10, 8))
x = np.linspace(-2,2,N)
plt.plot(x,f(x))
plt.show()
# -

# Create a noise version of the function
y = f(x) + np.random.randn(len(x))
plt.figure(figsize=(10, 8))
plt.plot(x,y)
plt.show()


# In many regression problems we have, or assume to have, knowledge about the underlying function. For example, we could assume that $f(x) = ax^2 + b\sin(\omega x)$ and try to find an optimal estimator for $a, b, \omega$. 
#
# On the other hand we may use a very general family of functions, like poynomials of order $p$,
# \begin{equation}
# \hat f(x) = a_0 + a_1 x + a_2 x^2 \ldots a_p x^p
# \end{equation}
# to approximate $f$. Let us see how this works out before using our neural networks.

# +
# Define estimator function. Arguments are inout variable, observation and polynomial order
# Returns a set of polynomial coefficients
def reg_estimator(x,y,p):
    # Use simple ls approach
    N = len(x)
    H = np.zeros((N,p+1))
    for col in range(p+1):
        H[:,col] = x**col
    iHtH = np.linalg.inv(np.dot(H.T,H))
    theta = np.dot(np.dot(iHtH,H.T),y)
    return theta

# Computes fx) = c_0 + c_1x + c_2 x^2 ... c_p x^p
def poly(x, C):
    # compute p(x) for coeffs in c
    y = 0*x
    for p, c in enumerate(C):
        y += c*x**p        
    return y


# -

# ## Problem 1
# Play with different $p$ to see how close you can get to the true function.
#
# Note: Very high $p$ will give numerical problems.

# +
# Give a set of polynomial orders to try
P = [1, 2, 5, 10, 20]

plt.figure(figsize=(10,8))
plt.plot(x,f(x),label="Truth")
for p in P:
    C = reg_estimator(x,y,p)
    plt.plot(x,poly(x,C),label="Poly order " + str(p))
plt.legend()
plt.show()


# -

# In what follows we will use a deep neural network to approximate $f$. We set this up below

# We train the network by using $x$ as an input and the squared error between the network output $\hat y$ and the observed value $y$ as a loss
# \begin{equation}
#  L = \frac{1}{N} \sum_n (\hat y - y)^2
# \end{equation}
#
# We first try our network on clean data to check if it works.

def run(x,y,M,n_layers):
    
    # Create the model 
    model = keras.Sequential()
    # Add first layer with dimension one
    model.add(keras.layers.Dense(M, activation=tf.nn.relu, input_dim=1))
    # Add more hidden layers
    for layer in range(n_layers-1):
        model.add(keras.layers.Dense(M, activation=tf.nn.relu))
    # Add output layer
    model.add(keras.layers.Dense(1))
    model.summary()

    # Train the model
    model.compile(loss='mean_squared_error',
                  optimizer="adam",
                  metrics=['accuracy'])
    
    history = model.fit(x, y, epochs=1000, batch_size=128, verbose=False)

    z = model.predict(x)
    plt.figure(figsize=(10,8))
    plt.plot(x,y,label="Truth")
    plt.plot(x,z,label="DNN")
    plt.legend()
    plt.show()


run(x,f(x),M=512,n_layers=1)

# ## Problem 2
# Try increasing the number of nodes in the network to see if the results can be improved.

run(x,f(x),M=1024,n_layers=1)

# Next we will use a deep network with more than one hidden layer.

run(x,f(x),M=16,n_layers=4)

# ## Problem 3
# Try increasing the number of hidden nodes per layer until performance is satisfactory. Can the same effect be achieved by just adding more layers?

run(x,f(x),M=32,n_layers=4)

run(x,f(x),M=64,n_layers=4)

run(x,f(x),M=16,n_layers=6)

run(x,f(x),M=16,n_layers=8)

run(x,f(x),M=16,n_layers=10)

# ## Problem 4
# Using the best setup from the previous problem, train a model using the noisy data.

y_noisy = f(x)+ np.random.randn(len(x))
run(x,y,M=16,n_layers=10)



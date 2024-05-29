# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: TTT4185
#     language: python
#     name: ttt4185
# ---

# # TTT4185 Machine learning for Speech technology
#
# ## Computer assigment 3b:  Regression analysis
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
# y = f(x) + n.
# \end{equation}
#
# We generate our data according to the function $f(x) = x^2 + \cos(20x) \text{ sign}(x)$, obtaining a set of observations $\{(x_i,y_i)\}$.
#
# Then we assume we do not know the underlying function and we try to recover and approximation of $f$ only using the observations $\{(x_i,y_i)\}$.

# +
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

import tensorflow as tf
from tensorflow import keras


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

# Create a noise version of the observations
y = f(x) + np.random.randn(len(x))
plt.figure(figsize=(10, 8))
plt.plot(x,y)
plt.show()

# One way to perform regression is to assume that the data is generated using a set of functions from a cerain family, for example polynomials of order $p$,
# \begin{equation}
# \hat f(x) = a_0 + a_1 x + a_2 x^2 \ldots a_p x^p.
# \end{equation}
# Then regression corresponds to fitting the parameters in the model. Let us see how this works out before using our neural networks.

# +
# Give a set of polynomial orders to try
P = [1, 2, 5, 10, 20]

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

plt.figure(figsize=(10,8))
plt.plot(x,f(x),label="Truth")
for p in P:
    C = reg_estimator(x,y,p)
    plt.plot(x,poly(x,C),label="Poly order " + str(p))
plt.legend()
plt.show()
# -

# ## Problem 1
# Play with different $p$ to see how close you can get to the true function.
#
# Note: Very high $p$ will give numerical problems.



# In what follows we will use a deep neural network to approximate $f$. We set this up below

# +
# Ceate a model with a single hidden layer. Note that input and output has
# dimension one
M = 512
model = keras.Sequential([
    keras.layers.Dense(M, activation=tf.nn.relu, input_dim=1),
    keras.layers.Dense(1)
])

model.summary()
# Train the model
model.compile(loss='mean_squared_error',
              optimizer="adam",
              metrics=['accuracy'])
# -

# We train the network by using $x$ as an input and the squared error between the network output $\hat y$ and the observed value $y$ as a loss
# \begin{equation}
#  L = \frac{1}{N} \sum_n (\hat y - y)^2
# \end{equation}
#
# We first try our network on clean data to check if it works.

# train the model
history = model.fit(x, f(x), epochs=1000, batch_size=128, verbose=True)

# Using the variable `history`, plot the evolution of the loss during training. 

# Compute \hat y from the network and compare this to the true f(x)
z = model.predict(x)
plt.figure(figsize=(10,8))
plt.plot(x,f(x),label="Truth")
plt.plot(x,z,label="DNN")
plt.legend()
plt.show()

# ## Problem 2
# Try increasing the number of nodes in the network to see if the results can be improved.



# Next we will use a deep network with more than one hidden layer.

# +
# Create a model with multiple hidden layers. Note that input and output has
# dimension one
M = 16
model = keras.Sequential([
    keras.layers.Dense(M, activation=tf.nn.relu, input_dim=1),
    keras.layers.Dense(M, activation=tf.nn.relu),
    keras.layers.Dense(M, activation=tf.nn.relu),
    keras.layers.Dense(M, activation=tf.nn.relu),
    keras.layers.Dense(1)
])
model.summary()

# Train the model
model.compile(loss='mean_squared_error',
              optimizer="adam",
              metrics=['accuracy'])

history = model.fit(x, f(x), epochs=1000, batch_size=128, verbose=True)
# -

z = model.predict(x)
plt.figure(figsize=(10,8))
plt.plot(x,f(x),label="Truth")
plt.plot(x,z,label="DNN")
plt.legend()
plt.show()

# ## Problem 3
# Try increasing the number of hidden nodes per layer until performance is satisfactory. Can the same effect be achieved by just adding more layers?



# ## Problem 4
# Using the best setup from the previous problem, train a model using the noisy data.



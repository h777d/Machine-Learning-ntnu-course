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
# ## Computer assignment 3a: Classification using Deep Neural Networks
#
# ## Suggested Solution

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# +
# Load data from CSV files
rawtrain = pd.read_csv("Train.csv")
rawval = pd.read_csv("Validation.csv")

# Take a peek at the raw data
rawtrain.head(10)

# +
# We will be classifying three different vowels. Extract the training and validation data
phonemes = ["ae", "ey", "ux"]
train = rawtrain[rawtrain["Phoneme"].isin(phonemes)]
val = rawval[rawval["Phoneme"].isin(phonemes)]
trainlabels = [phonemes.index(ph) for ph in train["Phoneme"]]
vallabels = [phonemes.index(ph) for ph in val["Phoneme"]]

# Fix labels. The "to_categorical" call maps integer labels {n}
# to a vector of length N (number of labels) with a one in position n
y_train = keras.utils.to_categorical(trainlabels, len(phonemes))
y_val = keras.utils.to_categorical(vallabels, len(phonemes))
# -

train.head()

y_train

# +
# Features to use
features = ["F1","F2"]

# Extract features
x_train_raw = train[features]
x_val_raw = val[features]

# Normalize to zero mean
x_mean = np.mean(x_train_raw)
x_std = np.std(x_train_raw)
x_train = x_train_raw - x_mean
x_val = x_val_raw - x_mean

# +
# Create a model with a single hidden layer
model = keras.Sequential([
    keras.layers.Dense(256, activation=tf.nn.relu, input_dim=x_train.shape[1]),
    keras.layers.Dense(len(phonemes), activation=tf.nn.softmax)
])
model.summary()

# Train the model
model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

# Set verbose=True to see more training information
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                    epochs=300, batch_size=32, verbose=False)
# -

# Visualize the training results
plt.figure(figsize=(10,8))
plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['accuracy'],label='acc')
plt.plot(history.history['val_loss'],label='val_loss')
plt.plot(history.history['val_accuracy'],label='val_acc')
plt.grid()
plt.legend()
plt.show()

# Evaluate model on the validation set
score = model.evaluate(x_val, y_val, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])

# ## Problem 1
# Increase the number of features to include "F3" and "F4" and rerun the experiments. Also try adding the bandwidths ("B1"-"B4").

# +
# Features to use
features = ["F1","F2","F3","F4"]

# Extract features
x_train_raw = train[features]
x_val_raw = val[features]

# Normalize to zero mean
x_mean = np.mean(x_train_raw)
x_std = np.std(x_train_raw)
x_train = x_train_raw - x_mean
x_val = x_val_raw - x_mean

# Create a model with a single hidden layer
model = keras.Sequential([
    keras.layers.Dense(256, activation=tf.nn.relu, input_dim=x_train.shape[1]),
    keras.layers.Dense(len(phonemes), activation=tf.nn.softmax)
])
model.summary()

# Train the model
model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

# Set verbose=True to see more training information
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                    epochs=300, batch_size=32, verbose=False)
# -

# Visualize the training results
plt.figure(figsize=(10,8))
plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['accuracy'],label='acc')
plt.plot(history.history['val_loss'],label='val_loss')
plt.plot(history.history['val_accuracy'],label='val_acc')
plt.grid()
plt.legend()
plt.show()

# Evaluate model on the validation set
score = model.evaluate(x_val, y_val, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])

# +
# Features to use
features = ["F1","F2","F3","F4","B1","B2","B3","B4"]

# Extract features
x_train_raw = train[features]
x_val_raw = val[features]

# Normalize to zero mean
x_mean = np.mean(x_train_raw)
x_std = np.std(x_train_raw)
x_train = x_train_raw - x_mean
x_val = x_val_raw - x_mean

# Create a model with a single hidden layer
model = keras.Sequential([
    keras.layers.Dense(256, activation=tf.nn.relu, input_dim=x_train.shape[1]),
    keras.layers.Dense(len(phonemes), activation=tf.nn.softmax)
])
model.summary()

# Train the model
model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

# Set verbose=True to see more training information
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                    epochs=300, batch_size=32, verbose=False)
# -

# Visualize the training results
plt.figure(figsize=(10,8))
plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['accuracy'],label='acc')
plt.plot(history.history['val_loss'],label='val_loss')
plt.plot(history.history['val_accuracy'],label='val_acc')
plt.grid()
plt.legend()
plt.show()

# Evaluate model on the validation set
score = model.evaluate(x_val, y_val, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])

# ## Problem 2
# Change the number of nodes in the hidden layer and see how the results change. Try using dropout, and observe the results.

# +
# Features to use
features = ["F1","F2","F3","F4"]

# Extract features
x_train_raw = train[features]
x_val_raw = val[features]

# Normalize to zero mean
x_mean = np.mean(x_train_raw)
x_std = np.std(x_train_raw)
x_train = x_train_raw - x_mean
x_val = x_val_raw - x_mean

# +
# Create a model with a single hidden layer
model = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.relu, input_dim=x_train.shape[1]),
    keras.layers.Dense(len(phonemes), activation=tf.nn.softmax)
])
model.summary()

# Train the model
model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

# Set verbose=True to see more training information
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                    epochs=300, batch_size=32, verbose=False)
# -

# Visualize the training results
plt.figure(figsize=(10,8))
plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['accuracy'],label='acc')
plt.plot(history.history['val_loss'],label='val_loss')
plt.plot(history.history['val_accuracy'],label='val_acc')
plt.grid()
plt.legend()
plt.show()

# Evaluate model on the validation set
score = model.evaluate(x_val, y_val, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])

# +
# Create a model with a single hidden layer
model = keras.Sequential([
    keras.layers.Dropout(rate=0.2, input_shape=(x_train.shape[1],)),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(len(phonemes), activation=tf.nn.softmax)
])
model.summary()

# Train the model
model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

# Set verbose=True to see more training information
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                    epochs=300, batch_size=32, verbose=False)
# -

# Visualize the training results
plt.figure(figsize=(10,8))
plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['accuracy'],label='acc')
plt.plot(history.history['val_loss'],label='val_loss')
plt.plot(history.history['val_accuracy'],label='val_acc')
plt.grid()
plt.legend()
plt.show()

# Evaluate model on the validation set
score = model.evaluate(x_val, y_val, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])

# +
# Create a model with a single hidden layer
model = keras.Sequential([
    keras.layers.Dropout(rate=0.2, input_shape=(x_train.shape[1],)),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(len(phonemes), activation=tf.nn.softmax)
])
model.summary()

# Train the model
model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

# Set verbose=True to see more training information
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                    epochs=300, batch_size=32, verbose=False)
# -

# Visualize the training results
plt.figure(figsize=(10,8))
plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['accuracy'],label='acc')
plt.plot(history.history['val_loss'],label='val_loss')
plt.plot(history.history['val_accuracy'],label='val_acc')
plt.grid()
plt.legend()
plt.show()

# Evaluate model on the validation set
score = model.evaluate(x_val, y_val, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])

# ## Problem 3
# Add multiple layers to the network and observe the results.

# +
# Create a model with a single hidden layer
model = keras.Sequential([
    keras.layers.Dropout(rate=0.2, input_shape=(x_train.shape[1],)),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(len(phonemes), activation=tf.nn.softmax)
])
model.summary()

# Train the model
model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

# Set verbose=True to see more training information
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                    epochs=300, batch_size=32, verbose=False)
# -

# Visualize the training results
plt.figure(figsize=(10,8))
plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['accuracy'],label='acc')
plt.plot(history.history['val_loss'],label='val_loss')
plt.plot(history.history['val_accuracy'],label='val_acc')
plt.grid()
plt.legend()
plt.show()

# Evaluate model on the validation set
score = model.evaluate(x_val, y_val, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])

# ## Problem 4
# Use the data to predict the gender of the speaker. Try including the format bandwidths as features as well ("B1"-"B4").

# +
# We will be classifying two different genders. Extract the training and validation data
gender = ["M", "F"]
train = rawtrain[rawtrain["Gender"].isin(gender)]
val = rawval[rawval["Gender"].isin(gender)]
trainlabels = [gender.index(gd) for gd in train["Gender"]]
vallabels = [gender.index(gd) for gd in val["Gender"]]
# Features to use
features = ["F1","F2","F3","F4"]

# Extract features
x_train_raw = train[features]
x_val_raw = val[features]

# Normalize to zero mean
x_mean = np.mean(x_train_raw)
x_std = np.std(x_train_raw)
x_train = x_train_raw - x_mean
x_val = x_val_raw - x_mean

# Fix labels. The "to_categorical" call maps integer labels {n}
# to a vector of length N (number of labels) with a one in position n
y_train = keras.utils.to_categorical(trainlabels, len(gender))
y_val = keras.utils.to_categorical(vallabels, len(gender))

# Create a model with a single hidden layer
model = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.relu, input_dim=x_train.shape[1]),
    keras.layers.Dense(len(gender), activation=tf.nn.softmax)
])
model.summary()

# Train the model
model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

# Set verbose=True to see more training information
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                    epochs=300, batch_size=32, verbose=False)

# +
# Visualize the training results
plt.figure(figsize=(10,8))
plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['accuracy'],label='acc')
plt.plot(history.history['val_loss'],label='val_loss')
plt.plot(history.history['val_accuracy'],label='val_acc')
plt.grid()
plt.legend()
plt.show()

# Evaluate model on the validation set
score = model.evaluate(x_val, y_val, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])
# -

# The result is ok but it can still be improved. Let's add more hidden layers and also the dropout layer

# +
model = keras.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu, input_dim=x_train.shape[1]),
    keras.layers.Dropout(rate=0.5, input_shape=(x_train.shape[1],)),
    keras.layers.Dense(256, activation=tf.nn.relu, input_dim=x_train.shape[1]),
    keras.layers.Dense(len(gender), activation=tf.nn.softmax)
])
model.summary()

# Train the model
model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

# Set verbose=True to see more training information
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                    epochs=300, batch_size=32, verbose=False)

# +
# Visualize the training results
plt.figure(figsize=(10,8))
plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['accuracy'],label='acc')
plt.plot(history.history['val_loss'],label='val_loss')
plt.plot(history.history['val_accuracy'],label='val_acc')
plt.grid()
plt.legend()
plt.show()

# Evaluate model on the validation set
score = model.evaluate(x_val, y_val, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])
# -

# Using more hidden units, layers and the dropout layer can slightly increase the accuracy for validation set.

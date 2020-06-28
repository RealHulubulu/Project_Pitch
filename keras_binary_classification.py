# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 20:19:55 2020

https://www.tensorflow.org/tutorials/structured_data/imbalanced_data

https://machinelearningmastery.com/how-to-prepare-categorical-data-for-deep-learning-in-python/#:~:text=Machine%20learning%20and%20deep%20learning,fit%20and%20evaluate%20a%20model.

"""

import tensorflow as tf
from tensorflow import keras

import os
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
#%%

def plot_cm(labels, predictions, p=0.5):
  cm = confusion_matrix(labels, predictions > p)
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix @{:.2f}'.format(p))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')

  print('<=50K Detected (True Negatives): ', cm[0][0])
  print('<=50K Incorrectly Detected (False Positives): ', cm[0][1])
  print('>50K Incorrectly Detected (False Negatives): ', cm[1][0])
  print('>50K Detected (True Positives): ', cm[1][1])
  print('Total Fraudulent Transactions: ', np.sum(cm[1]))
  
def prepare_inputs(data):
	oe = OrdinalEncoder()
	oe.fit(data)
	data_enc = oe.transform(data)
	return data_enc

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      ]

def make_model(metrics = METRICS, output_bias=None):
  if output_bias is not None:
    output_bias = tf.keras.initializers.Constant(output_bias)
  model = keras.Sequential([
      keras.layers.Dense(8, activation='relu', input_shape=(train_features.shape[-1],)),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),
  ])

  model.compile(
      optimizer=keras.optimizers.Adam(lr=1e-3),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics
      )

  return model


#%%
mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#%%

CSV_COLUMN_NAMES = ["age", "workclass", "fnlwgt", "education", "education-num", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", "capital-loss", "hours-per-week", "native-country", "income"] #adult.names
dftrain = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', names=CSV_COLUMN_NAMES, header=0)
dfeval = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test', names=CSV_COLUMN_NAMES, header=0)

dfeval['income'] = dfeval['income'].str.replace('.','')
dfeval['income'] = dfeval['income'].replace(' <=50K',0)
dfeval['income'] = dfeval['income'].replace(' >50K',1)

dftrain['income'] = dftrain['income'].replace(' <=50K',0)
dftrain['income'] = dftrain['income'].replace(' >50K',1)

label = "income"
frames = [dftrain,dfeval]
raw_df = pd.concat(frames)

raw_df.pop("education")
raw_df.pop('fnlwgt')

# print(dfeval['income'])
raw_df = raw_df.astype(str)

#ordinal encoding data
# prepare input data
# def prepare_inputs(data):
# 	oe = OrdinalEncoder()
# 	oe.fit(data)
# 	data_enc = oe.transform(data)
# 	return data_enc

ordinal_enc = prepare_inputs(raw_df)
# print(ordinal_enc)
# 0 is <=50k, 1 is >50K

raw_df = ordinal_enc

CSV_COLUMN_NAMES = ["age", "workclass", "education-num", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", "capital-loss", "hours-per-week", "native-country", "income"] #adult.names
raw_df = pd.DataFrame(raw_df, columns = CSV_COLUMN_NAMES)

neg, pos = np.bincount(raw_df['income'])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(total, pos, 100 * pos / total))

cleaned_df = raw_df.copy()

#%%
# Use a utility from sklearn to split and shuffle our dataset.
train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)

# Form np arrays of labels and features.
train_labels = np.array(train_df.pop('income')) # array of train labels

bool_train_labels = train_labels != 0

val_labels = np.array(val_df.pop('income')) # array of valid labels
test_labels = np.array(test_df.pop('income')) # array of test labels

train_features = np.array(train_df) # array train data
val_features = np.array(val_df) # array valid data
test_features = np.array(test_df) # array test data

scaler = StandardScaler() # used to normalize input, mean is 0, std dev is 1

train_features = scaler.fit_transform(train_features) 
# normalizes, fit here so model doesn't see valid or test

val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

train_features = np.clip(train_features, -5, 5)
val_features = np.clip(val_features, -5, 5)
test_features = np.clip(test_features, -5, 5)


print('Training labels shape:', train_labels.shape)
print('Validation labels shape:', val_labels.shape)
print('Test labels shape:', test_labels.shape)

print('Training features shape:', train_features.shape)
print('Validation features shape:', val_features.shape)
print('Test features shape:', test_features.shape)

#%%

# METRICS = [
#       keras.metrics.TruePositives(name='tp'),
#       keras.metrics.FalsePositives(name='fp'),
#       keras.metrics.TrueNegatives(name='tn'),
#       keras.metrics.FalseNegatives(name='fn'), 
#       keras.metrics.BinaryAccuracy(name='accuracy'),
#       keras.metrics.Precision(name='precision'),
#       keras.metrics.Recall(name='recall'),
#       keras.metrics.AUC(name='auc'),
# ]

# def make_model(metrics = METRICS, output_bias=None):
#   if output_bias is not None:
#     output_bias = tf.keras.initializers.Constant(output_bias)
#   model = keras.Sequential([
#       keras.layers.Dense(8, activation='relu', input_shape=(train_features.shape[-1],)),
#       keras.layers.Dropout(0.5),
#       keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),
#   ])

#   model.compile(
#       optimizer=keras.optimizers.Adam(lr=1e-3),
#       loss=keras.losses.BinaryCrossentropy(),
#       metrics=metrics
#       )

#   return model

EPOCHS = 500
BATCH_SIZE = 512

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

#%%

model = make_model()
model.summary() 

#%%

# model = make_model()
# # model.load_weights(initial_weights)
# baseline_history = model.fit(
#     train_features,
#     train_labels,
#     batch_size=BATCH_SIZE,
#     epochs=EPOCHS,
#     callbacks = [early_stopping],
#     validation_data=(val_features, val_labels))

#%%

# train_predictions_baseline = model.predict(train_features, batch_size=BATCH_SIZE)
# test_predictions_baseline = model.predict(test_features, batch_size=BATCH_SIZE)

# def plot_cm(labels, predictions, p=0.5):
#   cm = confusion_matrix(labels, predictions > p)
#   plt.figure(figsize=(5,5))
#   sns.heatmap(cm, annot=True, fmt="d")
#   plt.title('Confusion matrix @{:.2f}'.format(p))
#   plt.ylabel('Actual label')
#   plt.xlabel('Predicted label')

#   print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
#   print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
#   print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
#   print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
#   print('Total Fraudulent Transactions: ', np.sum(cm[1]))
  
# baseline_results = model.evaluate(test_features, test_labels,
#                                   batch_size=BATCH_SIZE, verbose=0)
# for name, value in zip(model.metrics_names, baseline_results):
#   print(name, ': ', value)
# print()

# plot_cm(test_labels, test_predictions_baseline)

#%%

# Add class weights

# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
weight_for_0 = (1 / neg)*(total)/2.0 
weight_for_1 = (1 / pos)*(total)/2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))

#%%

# Train Model using class weights

weighted_model = make_model()
# weighted_model.load_weights(initial_weights)

weighted_history = weighted_model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks = [early_stopping],
    validation_data=(val_features, val_labels),
    # The class weights go here
    class_weight=class_weight) 

#%%

train_predictions_weighted = weighted_model.predict(train_features, batch_size=BATCH_SIZE)
test_predictions_weighted = weighted_model.predict(test_features, batch_size=BATCH_SIZE)

weighted_results = weighted_model.evaluate(test_features, test_labels,
                                           batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(weighted_model.metrics_names, weighted_results):
  print(name, ': ', value)
print()

plot_cm(test_labels, test_predictions_weighted)

#%%
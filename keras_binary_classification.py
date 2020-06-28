# -*- coding: utf-8 -*-
"""
This is code that uses a Tensorflow tutorial on imbalanced datasets on the 
adult income dataset provided by UC Irvine. The positive class is >50K.

Link to dataset: https://archive.ics.uci.edu/ml/datasets/Adult
Link to tutorial: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
Link for categorical data handling: https://machinelearningmastery.com/how-to-prepare-categorical-data-for-deep-learning-in-python/#:~:text=Machine%20learning%20and%20deep%20learning,fit%20and%20evaluate%20a%20model.

"""
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
#%%

def plot_cm_show_metrics(labels, predictions, p=0.5):
    "creates a confusion matrix plot and prints tn,fp,fn,tp metrics"
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
    "function for transforming categorical data using OrdinalEncoder"
    oe = OrdinalEncoder()
	oe.fit(data)
	data_enc = oe.transform(data)
	return data_enc

def make_model(train_features, hidden_layer_size = 16, output_bias=None):
    "makes a model with fixed metrics"
    metrics = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'), 
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = keras.Sequential([
        keras.layers.Dense(hidden_layer_size, activation='relu', input_shape=(train_features.shape[-1],)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics
    )
    return model

def format_input(input_df, col_names):
    "input data pipeline with outputs train, valid, test, neg count, pos count"
    input_df = input_df.astype(str)
    encoded_df = prepare_inputs(input_df)
    encoded_df = pd.DataFrame(encoded_df, columns = col_names)
    
    neg, pos = np.bincount(encoded_df['income']) # neg = 0 (<=50k), pos = 1 (>50K)
    total = neg + pos
    print('Examples:\n Total: {}\n Positive: {} ({:.2f}% of total)\n Negative: {} ({:.2f}% of total)'.format(
    total, pos, 100 * pos/total, neg, 100 * neg/total))
    
    train_df, test_df = train_test_split(encoded_df, test_size=0.2)
    train_df, val_df = train_test_split(train_df, test_size=0.2)
    
    train_labels, val_labels, test_labels = np.array(train_df.pop('income')), np.array(val_df.pop('income')), np.array(test_df.pop('income')) # array of train labels    
    train_features, val_features, test_features = np.array(train_df), np.array(val_df), np.array(test_df) # array train data
    
    scaler = StandardScaler() # used to normalize input, mean is 0, std dev is 1
    
    train_features = scaler.fit_transform(train_features) # fits here so model doesn't see val or test
    val_features, test_features = scaler.transform(val_features), scaler.transform(test_features)
    
    train_features, val_features, test_features = np.clip(train_features, -5, 5), np.clip(val_features, -5, 5), np.clip(test_features, -5, 5)
    
    return train_features, val_features, test_features, train_labels, val_labels, test_labels, neg, pos

def create_class_weights(neg, pos):
    "creates class weights for the model"
    total = neg + pos
    weight_for_0 = (1 / neg)*(total)/2.0 
    weight_for_1 = (1 / pos)*(total)/2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print('Weight for class 0: {:.2f}'.format(weight_for_0)) # 0.66
    print('Weight for class 1: {:.2f}'.format(weight_for_1)) # 2.09
    return class_weight

def train_model(weighted_model, train_features, train_labels, EPOCHS, BATCH_SIZE, 
                val_features, val_labels, class_weight = None):
    "trains the model using fixed early_stopping with output of model history"
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc', 
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True
    )
    weighted_history = weighted_model.fit(
        train_features,
        train_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks = [early_stopping],
        validation_data=(val_features, val_labels),
        class_weight=class_weight
    ) 
    return weighted_history

def eval_model_show_metrics(weighted_model):
    "evaluates the model, prints metrics, outputs model results"
    weighted_results = weighted_model.evaluate(test_features, test_labels,
                                           batch_size=BATCH_SIZE, verbose=0)
    for name, value in zip(weighted_model.metrics_names, weighted_results):
        print(name, ': ', value)
    return weighted_results

def pred_model_show_confusion_matrix(weighted_model):
    "predicts using the model, displays confusion matrix, outputs model predictions"
    # train_predictions_weighted = weighted_model.predict(train_features, batch_size=BATCH_SIZE)
    test_predictions_weighted = weighted_model.predict(test_features, batch_size=BATCH_SIZE)
    plot_cm_show_metrics(test_labels, test_predictions_weighted)
    return test_predictions_weighted

#%%
# Prepare data, declare global colors

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

CSV_COLUMN_NAMES = ["age", "workclass", "fnlwgt", "education", "education-num", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", "capital-loss", "hours-per-week", "native-country", "income"] #adult.names
dftrain = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', names=CSV_COLUMN_NAMES, header=0)
dfeval = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test', names=CSV_COLUMN_NAMES, header=0)
label = "income"
dfeval['income'] = dfeval['income'].str.replace('.','')
frames = [dftrain,dfeval]
raw_df = pd.concat(frames)
raw_df.pop("education")
raw_df.pop('fnlwgt')
CSV_COLUMN_NAMES.remove('education')
CSV_COLUMN_NAMES.remove('fnlwgt')

#%%
# Input data pipeline, creates model

EPOCHS = 500
BATCH_SIZE = 512
train_features, val_features, test_features, train_labels, val_labels, test_labels, neg, pos = format_input(raw_df, CSV_COLUMN_NAMES)
class_weight = create_class_weights(neg, pos)
weighted_model = make_model(train_features, hidden_layer_size = 16)
weighted_model.summary()
#%%
# Train model

weighted_history = train_model(weighted_model, train_features, train_labels, EPOCHS, BATCH_SIZE,
                               val_features, val_labels, class_weight = class_weight)
#%%
# Results and predictions

weighted_results = eval_model_show_metrics(weighted_model)
test_predictions_weighted = pred_model_show_confusion_matrix(weighted_model)


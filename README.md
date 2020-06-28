# Project_Pitch

This is a simple model that can do binary classification on an imbalanced dataset. It uses the basic
example code from https://www.tensorflow.org/tutorials/structured_data/imbalanced_data on the adult
dataset from UC Irvine found at https://archive.ics.uci.edu/ml/datasets/Adult . This is an
imbalanced dataset with about a 76%/24% split between the two classes represented. 

**The metrics and info from training the model using weights for the classes are below.**

Positive class: >60K
Negative class: <=50k

loss :  0.4900376870976535

tp :  1781.0

fp :  1911.0

tn :  5553.0

fn :  524.0

accuracy :  0.75074214

precision :  0.48239437

recall :  0.7726681

auc :  0.8448659

**Below is the confusion matrix made from this model. Positive class is 1. Negative class is 0.**

![confusion_matrix](/confusion_matrix_example.png)

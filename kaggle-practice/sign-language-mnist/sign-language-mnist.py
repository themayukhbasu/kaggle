import tensorflow as tf
import numpy as np
import pandas as pd

print(tf.__version__)

train_data = pd.read_csv('data/sign_mnist_train.csv')
test_data = pd.read_csv('data/sign_mnist_test.csv')
# print(train_data.head())

train_x = train_data
train_y = train_x.pop('label')

print(train_x.head())
import csv
import tensorflow as tf
import numpy as np

def get_data(fname):
    images, labels = [], []

    with open(fname) as f:
        file = csv.reader(f)
        next(file) # skip header row
        for row in file:
            row = np.array(row).astype(float)
            label = row[0]
            image = np.reshape(row[1:], (28,28))
            images.append(image)
            labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    print("Found %d images" % images.shape[0])
    return images, labels

get_data('data/sign_mnist_train.csv')
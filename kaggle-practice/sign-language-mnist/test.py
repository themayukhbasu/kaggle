import tensorflow as tf
import numpy as np
import pandas as pd

print(tf.__version__)
print()

def test1():
    df_train_data = pd.read_csv('data/sign_mnist_train.csv')
    df_test_data = pd.read_csv('data/sign_mnist_test.csv')
    # print(train_data.head())

    df_train_x = df_train_data
    df_train_y = df_train_x.pop('label')


    df_test_x = df_test_data
    df_test_y = df_test_x.pop('label')

    print(df_train_x.head())
    print(type(df_train_x))
    print(type(df_train_x.values))

    train_ds = tf.data.Dataset.from_tensor_slices((df_train_x.values, df_train_y.values))
    test_ds = tf.data.Dataset.from_tensor_slices((df_test_x.values, df_test_y.values))


    def display_sample_data(dataset, flag=True):
        # displays 1 sample from dataset
        # 2 different ways to do this
        # you need to put the dataset in some sort of iterator
        if flag:
            for x, y in dataset.take(1):  # for in -> puts the dataset in an iterator
                print(x, "\n", y)
        else:
            x, y = next(iter(dataset))  # next() & iter() -> puts the dataset in an interator and then gets the next value
            print(x, "\n", y)


    def prep_dataset(dataset, batch_size=32):
        dataset = dataset.map(lambda x, y: tf.reshape(x, (28,28)))
        dataset = dataset.batch(batch_size)
        dataset = dataset.cache().prefetch(1)
        return dataset
    #dataset = prep_dataset(train_ds)
    display_sample_data(train_ds)

def test2():
    print("Test 2 start")
    url ="https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip"
    fname="horse-or-human.zip"
    parent_dataset_dir = 'data'
    dataset_dir = 'horses_humans'
    tf.keras.utils.get_file(
            fname=fname,
            origin=url,
            #untar=True,
            cache_dir=parent_dataset_dir,
            cache_subdir=dataset_dir,
            extract=True
        )
    print("Test 2 end")

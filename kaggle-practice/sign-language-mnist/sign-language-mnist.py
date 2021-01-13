import csv
import tensorflow as tf
import numpy as np


def get_data(fname):
    images, labels = [], []

    with open(fname) as f:
        file = csv.reader(f)
        next(file)  # skip header row
        for row in file:
            row = np.array(row).astype(float)
            label = row[0]
            image = np.reshape(row[1:], (28, 28))
            images.append(image)
            labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    print("Found %d images" % images.shape[0])
    return images, labels


train_x, train_y = get_data('data/sign_mnist_train.csv')
test_x, test_y = get_data('data/sign_mnist_test.csv')

train_x = np.expand_dims(train_x, axis=-1)
test_x = np.expand_dims(test_x, axis=-1)


batch_size =32
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale= 1./255.0,
    horizontal_flip=True,
    vertical_flip=True,
    #zoom_range=0.2,
    #shear_range=0.2,
    validation_split=0.2
)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255.0
)
train_data_generator = train_datagen.flow(train_x, train_y, subset='training')
val_data_generator = train_datagen.flow(train_x, train_y, batch_size=batch_size, subset='validation')
test_data_generator = test_datagen.flow(test_x, test_y, batch_size=batch_size)

class myEarlyStopCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy')>0.98:
            print("Model has reached desired training accuracy")
            self.model.stop_training=True

def simple_feed_forward(train_datagen, test_datagen, train_x, train_y, test_x, test_y):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28,1)),
        tf.keras.layers.Dense(128, activation='relu'),
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(26, activation='softmax') # classes integers range from 0 to 25
    ])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics='accuracy')
    epochs = 10
    history = model.fit(
        train_datagen.flow(train_x, train_y, batch_size=32, subset='training'),
        validation_data=train_datagen.flow(train_x, train_y, batch_size=32, subset='validation'),
        epochs=epochs
    )

    loss, acc = model.evaluate(test_datagen.flow(test_x, test_y))

    print("Validation Loss: ",loss, ' and accuracy: ', acc)

def using_cnn(train_data_generator, val_data_generator, test_data_generator):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=[28, 28, 1]),
        tf.keras.layers.Conv2D(32,3, activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(26, activation='softmax')
    ])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer='adam',
                  metrics='accuracy')
    my_early_stopping_cb = myEarlyStopCallback()

    model.fit(train_data_generator,
              validation_data=val_data_generator,
              epochs=10,
              callbacks=[my_early_stopping_cb])
    loss, acc = model.evaluate(test_data_generator)
    print("Test loss is %f and accuracy is %f" % (loss, acc))


#simple_feed_forward(train_datagen, test_datagen, train_x, train_y, test_x, test_y)
using_cnn(train_datagen, test_datagen, train_x, train_y, test_x, test_y)
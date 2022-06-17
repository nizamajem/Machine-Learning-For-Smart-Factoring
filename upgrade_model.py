
#import standart modeule
import math
import numpy as np
import pandas as pd

import scikitplot
import seaborn as sns
from matplotlib import pyplot

#import sklearn module
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

#import tensorflow module
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, Activation
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.utils import np_utils


def augment_pixels(px, IMG_SIZE=48):
    image = np.array(px.split(' ')).reshape(IMG_SIZE, IMG_SIZE).astype('float32')
    image = tf.image.random_flip_left_right(image.reshape(IMG_SIZE, IMG_SIZE, 1))
    # Pad image size
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 12, IMG_SIZE + 12)
    # Random crop back to the original size
    image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 1])
    image = tf.image.random_brightness(image, max_delta=0.5)  # Random brightness
    image = tf.clip_by_value(image, 0, 255)
    augmented = image.numpy().reshape(IMG_SIZE, IMG_SIZE)
    str_augmented = ' '.join(augmented.reshape(IMG_SIZE * IMG_SIZE).astype('int').astype(str))
    return str_augmented
def deployment ():
    # import dataset
    df = pd.read_csv('fer2013.csv')
    df.tail(10)

    # create 7 primary classes for the emotion
    emotion_label_to_text = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happiness', 4: 'sadness', 5: 'surprise',
                             6: 'neutral'}

    # check visualization from values dataset
    sns.countplot(df.emotion)
    pyplot.show()

    # create augmentation function for fix unbalance dataset
    valcounts = df.emotion.value_counts()
    valcounts_diff = valcounts[valcounts.idxmax()] - valcounts
    for emotion_idx, aug_count in valcounts_diff.iteritems():
        sampled = df.query("emotion==@emotion_idx").sample(aug_count, replace=True)
        sampled['pixels'] = sampled.pixels.apply(augment_pixels)
        df = pd.concat([df, sampled])
        print(emotion_idx, aug_count)

    # Check again to see if dataset size is similar across emotions.
    sns.countplot(df.emotion)
    pyplot.show()

    # Make intereted labels using array
    INTERESTED_LABELS = [0, 1, 2, 3, 4, 5, 6]

    df = df[df.emotion.isin(INTERESTED_LABELS)]

    # convert to array every image and make it become stack
    img_array = df.pixels.apply(lambda x: np.array(x.split(' ')).reshape(48, 48, 1).astype('float32'))
    img_array = np.stack(img_array, axis=0)

    # define the LableEncoder funtion for convert every classess in emotion become the true labels
    le = LabelEncoder()
    img_labels = le.fit_transform(df.emotion)
    img_labels = np_utils.to_categorical(img_labels)
    img_labels.shape

    # split dataset into train and validation datsaet with test size 10%
    X_train, X_valid, y_train, y_valid = train_test_split(img_array, img_labels,
                                                          shuffle=True, stratify=img_labels,
                                                          test_size=0.1, random_state=42)
    # let's see every shape aftre spliting
    X_train.shape, X_valid.shape, y_train.shape, y_valid.shape

    # define varibale for taking a size of image so we'll got (48,48) with 1 dimension and 7 with classes
    img_width = X_train.shape[1]
    img_height = X_train.shape[2]
    img_depth = X_train.shape[3]
    num_classes = y_train.shape[1]

    # define model arsitektur with 4 convolusi layers and 3 dense layers
    model = Sequential([
        Conv2D(64, (5, 5), input_shape=(img_width, img_height, img_depth), activation='elu', padding='same',
               kernel_initializer='he_normal'),
        BatchNormalization(),
        Conv2D(64, (5, 5), activation='elu', padding='same', kernel_initializer='he_normal'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.4),
        Conv2D(128, (3, 3), activation='elu', padding='same', kernel_initializer='he_normal'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='elu', padding='same', kernel_initializer='he_normal'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.4),
        Conv2D(256, (3, 3), activation='elu', padding='same', kernel_initializer='he_normal'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='elu', padding='same', kernel_initializer='he_normal'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.5),
        Flatten(),
        Dense(64, activation='elu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.6),
        Dense(128, activation='elu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.6),
        Dense(num_classes, activation='softmax')
    ])

    # As the data in hand is less as compared to the task so ImageDataGenerator is good to go.
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
    )
    train_datagen.fit(X_train)

    batch_size = 32  # batch size of 32 performs the best.
    epochs = 10
    # I tried both `Nadam` and `Adam`, the difference in results is not different but I finally went with Nadam as it is more popular.
    optims = [
        optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam'),
        optimizers.Adam(0.001),
    ]
    # compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optims[1],
        metrics=['accuracy']
    )
    # and let's train the model
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_valid, y_valid),
        steps_per_epoch=len(X_train) / batch_size,
        epochs=10
    )
    return model

if __name__ == '__main__':
    model = deployment()
    model.save("upgrade_model.h5")
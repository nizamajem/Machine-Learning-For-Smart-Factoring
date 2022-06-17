
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback
from keras.callbacks import EarlyStopping

class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.95 and logs.get('loss')<0.001):
            print("\nAkurasi telah mencapai > 95%! dan loss < 0.001")
            self.model.stop_training = True

callbacks = myCallback()
def deployment():
    traindir = 'dataset/train'
    label = []
    classes = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_dataset = train_datagen.flow_from_directory(batch_size=64,
                                                      directory='dataset/imagesFear2013/train',
                                                      shuffle=True,
                                                      target_size=(48, 48),
                                                      subset="training",
                                                      color_mode="grayscale",
                                                      class_mode='categorical',
                                                      classes=classes
                                                      )
    validation_dataset = train_datagen.flow_from_directory(batch_size=64,
                                                           directory='dataset/imagesFear2013/test',
                                                           shuffle=True,
                                                           target_size=(48, 48),
                                                           subset="validation",
                                                           color_mode="grayscale",
                                                           class_mode='categorical',
                                                           classes=classes
                                                           )

    checkpointLoss = ModelCheckpoint(f"{modelFolder}bestLoss.hdf5",
                                     monitor='loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='auto')

    checkpointValLoss = ModelCheckpoint(f"{modelFolder}bestValLoss.hdf5",
                                        monitor='val_loss',
                                        verbose=1,
                                        save_best_only=True,
                                        mode='auto')

    earlyStopVal = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    earlyStop = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)

    """model = tf.keras.models.Sequential([
        # convolution 1
        tf.keras.layers.Conv2D(64, (4, 4), activation = 'relu', input_shape = (48, 48, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # convolution 2
        tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
        tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
        tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2,2)),
        # concolution 3
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.AveragePooling2D(pool_size=(3,3), strides=(2,2)),
        tf.keras.layers.Flatten(),
        # neural network
        tf.keras.layers.Dense(1024, activation = 'relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(1024, activation = 'relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(7, activation = 'softmax')
    ])"""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (4, 4), padding='same', kernel_regularizer=regularizers.l2(1e-4),
                               input_shape=(48, 48, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (4, 4,), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(rate=0.3),

        tf.keras.layers.Conv2D(64, (4, 4,), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (4, 4,), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(rate=0.4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='linear'),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer='adagrad',
                  metrics=['accuracy'])
    history = model.fit(train_dataset,
                        steps_per_epoch=100,
                        epochs=100,
                        validation_data=validation_dataset,
                        validation_steps=5,
                        callbacks=[callbacks,
                                   checkpointValLoss,
                                   earlyStopVal])
    # plot akurasi model
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuuracy Model')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.show()

    # plot loss model
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss Model')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    return model
if __name__ == '__main__':
    model = deployment()
    model.save("upgrade_model.h5")
# -*- coding: utf-8 -*-
# @Brief : cnn model training code, the training code will be saved in the models directory,
# and the line graph will be saved in the results directory
# Convolutional neural network based on LeNet-5 two-layer convolution and two-layer pooling layer

import tensorflow as tf
import matplotlib.pyplot as plt
from time import *


# Dataset loading function,
# Indicate the location of the data set and uniformly process it as the size of imgheight*imgwidth,
# Set batch at the same time
def data_load(data_dir, test_data_dir, img_height, img_width, batch_size):
    # load the training set
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    # load test set
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names
    # Return the processed training set, validation set and class name
    return train_ds, val_ds, class_names


def model_load(IMG_SHAPE=(224, 224, 3), class_num=12):

    model = tf.keras.models.Sequential([
        # Normalize the model, and uniformly process the numbers between 0-255 to between 0 and 1
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=IMG_SHAPE),
        # FIXME level one
        # Convolutional layer, the output of the convolutional layer is 32 channels,
        # the size of the convolution kernel is 3*3, and the activation function is relu
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        # Add a pooling layer, the pooling kernel size is 2*2
        tf.keras.layers.MaxPooling2D(2, 2),
        # FIXME Second floor
        # Convolution layer, the output is 64 channels,
        # the convolution kernel size is 3*3, and the activation function is relu
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # Pooling layer, maximum pooling, pooling operation on 2*2 areas
        tf.keras.layers.MaxPooling2D(2, 2),
        # Randomly discard part of the neuron output, regularize, reduce overfitting
        tf.keras.layers.Dropout(0.2),
        # Convert 2D output to 1D
        tf.keras.layers.Flatten(),
        # Same 128 dense layers and 10 output layers as the pre-convolution example
        tf.keras.layers.Dense(128, activation='relu'),
        # Through the softmax function, the model is output as a neuron with the length of the class name,
        # and the activation function uses the corresponding probability value of softmax
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])
    # output model information
    model.summary()
    # Indicate the training parameters of the model, the optimizer is the sgd optimizer,
    # and the loss function is the cross-entropy loss function
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Curves showing the training process
def show_loss_acc(history):
    # Extract model training set and verification set accuracy information and error information from history
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Output the picture according to the upper and lower structure
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Improved LeNet-5 Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Improved LeNet-5 Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('results/results_cnn.png', dpi=100)
    plt.close()


def train(epochs):
    # Start training, record start time
    begin_time = time()
    # todo load dataset
    train_ds, val_ds, class_names = data_load("../data/train",
                                              "../data/test", 224, 224, 16)
    print(class_names)
    # load model
    model = model_load(class_num=len(class_names))
    # Indicate the number of rounds of training epoch, start training
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    # todo save model
    model.save("models/cnn_fv.h5")
    # Record end time
    end_time = time()
    run_time = end_time - begin_time
    print('The cyclic program run time:', run_time, "s")  # The console prints the running time of the loop program
    # Draw a diagram of the model training process
    show_loss_acc(history)


if __name__ == '__main__':
    train(epochs=30)

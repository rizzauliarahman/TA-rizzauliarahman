import os, sys

home = os.path.dirname(os.getcwd())
sys.path.insert(0, home + '\\Preprocess')

import keras
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD, Adamax, Nadam
from sklearn.model_selection import StratifiedKFold
import os
import pickle
import methods as mt
import matplotlib.pyplot as plt
import time

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def CNN_model_1(filter_size: int):
    model = Sequential()

    model.add(Conv2D(8, (filter_size, filter_size), activation='relu', input_shape=(96, 96, 3)))
    model.add(Conv2D(16, (filter_size, filter_size), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (filter_size, filter_size), activation='relu'))
    model.add(Conv2D(64, (filter_size, filter_size), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(3, activation='softmax'))

    return model


def CNN_model_2(filter_size: int):
    model = Sequential()

    model.add(Conv2D(8, (filter_size, filter_size), activation='relu', input_shape=(96, 96, 3)))
    model.add(Conv2D(16, (filter_size, filter_size), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(3, activation='softmax'))

    return model


def CNN_model_3(filter_size: int):
    model = Sequential()

    model.add(Conv2D(8, (filter_size, filter_size), activation='relu', input_shape=(96, 96, 3)))
    model.add(Conv2D(16, (filter_size, filter_size), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (filter_size, filter_size), activation='relu'))
    model.add(Conv2D(64, (filter_size, filter_size), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (filter_size, filter_size), activation='relu'))
    model.add(Conv2D(64, (filter_size, filter_size), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(3, activation='softmax'))

    return model


def CNN_model_4(filter_size: int):
    model = Sequential()

    model.add(Conv2D(4, (filter_size, filter_size), activation='relu', input_shape=(96, 96, 3)))
    model.add(Conv2D(8, (filter_size, filter_size), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(16, (filter_size, filter_size), activation='relu'))
    model.add(Conv2D(32, (filter_size, filter_size), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(3, activation='softmax'))

    return model


def CNN_model_5(filter_size: int):
    model = Sequential()

    model.add(Conv2D(16, (filter_size, filter_size), activation='relu', input_shape=(96, 96, 3)))
    model.add(Conv2D(32, (filter_size, filter_size), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (filter_size, filter_size), activation='relu'))
    model.add(Conv2D(128, (filter_size, filter_size), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(3, activation='softmax'))

    return model


def train_model(dataset, model, optimizer: int, idx_model: int):
    home = os.path.dirname(os.getcwd())
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    x_train = np.array([data[0] for data in dataset])
    y_train = [data[1] for data in dataset]

    fopen = open(home + "\\Attributes\\race_list.dat", mode='rb')
    l_name = pickle.load(fopen)
    fopen.close()

    y_train = [l_name.index(race) for race in y_train]

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    csvscores = []
    n_trains = []
    n_valids = []
    n = 1

    optimizers = ['Adam', 'SGD', 'Adamax', 'Nadam']

    start = time.time()

    for train, val in kfold.split(x_train, y_train):
        y_train_sp = np.array(keras.utils.to_categorical(y_train, num_classes=len(l_name)))

        print('Fold - %d' % n)
        model.compile(optimizer=optimizers[optimizer], loss='categorical_crossentropy',
                      metrics=['accuracy', 'categorical_accuracy'])

        model.fit(x_train[train], y_train_sp[train], epochs=10, batch_size=20, verbose=1)

        scores = model.evaluate(x_train[val], y_train_sp[val], verbose=1)
        print("Fold Accuracy : %.2f%%\n" % (scores[1] * 100))
        csvscores.append(scores[1] * 100)
        n_trains.append(len(y_train_sp[train]))
        n_valids.append(len(y_train_sp[val]))
        n += 1

    diff = time.time() - start
    hour = diff // 3600
    exc = diff % 3600
    minutes = exc // 60
    seconds = exc % 60

    filename = "CNN_scenario%d.h5" % idx_model
    model.save(home + "\\Attributes\\" + filename)

    print("Average Folds Accuracy %.2f%% (+/- %.2f%%)" % (np.mean(csvscores), np.std(csvscores)))

    txtopen = open(home + "\\Training Results\\cnn_model_" + repr(idx_model) + ".txt", mode="w")
    txtopen.write("\nCNN MODEL " + repr(idx_model) + "\n")
    txtopen.write("\nTRAINING PERFORMANCE\n")

    txtopen.write("Number of folds: 10\n")
    txtopen.write("Number of epochs: 10\n")
    txtopen.write("Batch size: 20\n")
    txtopen.write("Optimizer: %s\n" % optimizers[optimizer])
    txtopen.write("Time elapsed for training: %d hour(s), %d minute(s), %d second(s)\n" % (hour, minutes, seconds))

    txtopen.write("\nFolds Performance:\n")
    for i in range(len(csvscores)):
        txtopen.write("Fold-%d, %d Training data, %d Validation data, Accuracy: %.2f%%\n" % (
        (i + 1), n_trains[i], n_valids[i], csvscores[i]))

    txtopen.write("\nAverage Folds Accuracy %.2f%% (+/- %.2f%%)" % (np.mean(csvscores), np.std(csvscores)))
    txtopen.close()


def test_model(dataset, model, idx_model):
    home = os.path.dirname(os.getcwd())
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    x_test = np.array([data[0] for data in dataset])
    y_test = [data[1] for data in dataset]

    fopen = open(home + "\\Attributes\\race_list.dat", mode='rb')
    l_name = pickle.load(fopen)
    fopen.close()

    txtopen = open(home + "\\Results\\cnn_model_" + repr(idx_model) + ".txt", mode='w')

    y_test = [l_name.index(race) for race in y_test]

    classes = model.predict(x_test, verbose=0)

    acc = mt.count_accuracy(classes, y_test)

    txtopen.write("\nCNN MODEL " + repr(idx_model) + "\n")
    model.summary(print_fn=lambda x: txtopen.write(x + '\n'))
    txtopen.write("\n=================================================================\n")

    mt.confusion_matrix(classes, y_test, l_name, txtopen)

    txtopen.write("\nModel accuracy: %.3f%%\n\n" % (acc * 100))
    txtopen.close()


def test_img(img, model, idx_model):
    home = os.path.dirname(os.getcwd())
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    plt.imshow(img)
    plt.show()

    x_test = np.array([img])

    fopen = open(home + "\\Attributes\\race_list.dat", mode='rb')
    l_name = pickle.load(fopen)
    fopen.close()

    classes = model.predict(x_test, verbose=0)

    print(l_name[np.argmax(classes[0])])

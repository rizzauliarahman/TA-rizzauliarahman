import os, sys

home = os.path.dirname(os.getcwd())
sys.path.insert(0, home + '\\Preprocess')

import keras
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD, Adamax, Nadam
from sklearn.model_selection import StratifiedKFold
import os
import pickle
import methods as mt


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


def train_model(dataset, model, optimizer: int):
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
    n = 1

    optimizers = ['Adam', 'SGD', 'Adamax', 'Nadam']

    for train, val in kfold.split(x_train, y_train):
        y_train_sp = np.array(keras.utils.to_categorical(y_train, num_classes=len(l_name)))

        print('Fold - %d' % n)
        model.compile(optimizer=optimizers[optimizer], loss='categorical_crossentropy',
                      metrics=['accuracy', 'categorical_accuracy'])

        model.fit(x_train[train], y_train_sp[train], epochs=10, batch_size=20, verbose=1)

        scores = model.evaluate(x_train[val], y_train_sp[val], verbose=1)
        print("Fold Accuracy : %.2f%%\n" % (scores[1] * 100))
        csvscores.append(scores[1] * 100)
        n += 1

    model.save('coba_model.h5')

    print("Average Folds Accuracy %.2f%% (+/- %.2f%%)" % (np.mean(csvscores), np.std(csvscores)))


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

    txtopen.write("\nCNN MODEL" + repr(idx_model) + "\n")
    model.summary(print_fn=lambda x: txtopen.write(x + '\n'))
    txtopen.write("\n=================================================================\n")
    txtopen.write("=================================================================\n")

    txtopen.write("\nModel accuracy: %.3f%%\n\n" % (acc * 100))

    mt.confusion_matrix(classes, y_test, l_name, txtopen)

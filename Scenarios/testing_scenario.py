import os, sys

home = os.path.dirname(os.getcwd())
sys.path.insert(0, home + '\\Preprocess')

import load_train_images as lti
import methods as mt
import pickle
import random
import classifier as cl
import keras


def scenario1():
    home = os.path.dirname(os.getcwd())
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    fopen = open(home + "\\Attributes\\test_dataset.dat", mode='rb')
    test = pickle.load(fopen)
    fopen.close()

    model = keras.models.load_model(home + "\\Attributes\\CNN_scenario1.h5")
    cl.test_model(test, model, 1)


def scenario2():
    home = os.path.dirname(os.getcwd())
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    fopen = open(home + "\\Attributes\\test_dataset.dat", mode='rb')
    test = pickle.load(fopen)
    fopen.close()

    test = mt.convert_to_grayscale(test)

    model = keras.models.load_model(home + "\\Attributes\\CNN_scenario2.h5")
    cl.test_model(test, model, 2)


if __name__ == '__main__':
    scenario1()

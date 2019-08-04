import os, sys

home = os.path.dirname(os.getcwd())
sys.path.insert(0, home + '\\Preprocess')

import load_train_images as lti
import methods as mt
import pickle
import random
import classifier as cl
import keras
from PIL import Image
import numpy as np


def scenario1():
    home = os.path.dirname(os.getcwd())
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    fopen = open(home + "\\Attributes\\test_dataset.dat", mode='rb')
    test = pickle.load(fopen)
    fopen.close()

    model = keras.models.load_model(home + "\\Attributes\\CNN_scenario4.h5")
    cl.test_model(test, model, 4)
    # img = Image.open("3257.jpg")
    # img = img.resize((96, 96))
    # img = np.array(img)
    # cl.test_img(img, model, 1)


def scenario2():
    home = os.path.dirname(os.getcwd())
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    fopen = open(home + "\\Attributes\\test_dataset.dat", mode='rb')
    test = pickle.load(fopen)
    fopen.close()

    test = mt.convert_to_grayscale(test)

    model = keras.models.load_model(home + "\\Attributes\\CNN_scenario2.h5")
    cl.test_model(test, model, 2)


def scenario5():
    home = os.path.dirname(os.getcwd())
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    fopen = open(home + "\\Attributes\\test_dataset.dat", mode='rb')
    test = pickle.load(fopen)
    fopen.close()

    model = keras.models.load_model(home + "\\Attributes\\CNN_scenario5.h5")
    cl.test_model(test, model, 5)


def scenario6():
    home = os.path.dirname(os.getcwd())
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    fopen = open(home + "\\Attributes\\test_dataset.dat", mode='rb')
    test = pickle.load(fopen)
    fopen.close()

    model = keras.models.load_model(home + "\\Attributes\\CNN_scenario6.h5")
    cl.test_model(test, model, 6)


def scenario7():
    home = os.path.dirname(os.getcwd())
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    fopen = open(home + "\\Attributes\\test_dataset.dat", mode='rb')
    test = pickle.load(fopen)
    fopen.close()

    model = keras.models.load_model(home + "\\Attributes\\CNN_scenario7.h5")
    cl.test_model(test, model, 7)


def scenario8():
    home = os.path.dirname(os.getcwd())
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    fopen = open(home + "\\Attributes\\test_dataset.dat", mode='rb')
    test = pickle.load(fopen)
    fopen.close()

    model = keras.models.load_model(home + "\\Attributes\\CNN_scenario8.h5")
    cl.test_model(test, model, 8)


if __name__ == '__main__':
    scenario8()

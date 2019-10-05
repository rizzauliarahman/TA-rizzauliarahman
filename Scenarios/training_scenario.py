import os, sys
home = os.path.dirname(os.getcwd())
sys.path.insert(0, home + '\\Preprocess')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import load_train_images as lti
import methods as mt
import pickle
import random
import classifier as cl
import time


def create_dataset():
    imgs = lti.load_all_train_images()

    home = os.path.dirname(os.getcwd())
    fopen = open(home + "\\Attributes\\races.dat", mode='rb')
    races = pickle.load(fopen)
    fopen.close()

    dataset = []

    for img, race in zip(imgs, races):
        dataset.append([img, race])

    random.shuffle(dataset)

    train, test = mt.split_test_train(dataset)

    fopen = open(home + "\\Attributes\\train_dataset.dat", mode='wb')
    pickle.dump(train, fopen)
    fopen.close()

    fopen = open(home + "\\Attributes\\test_dataset.dat", mode='wb')
    pickle.dump(test, fopen)
    fopen.close()

    race_list = ['Caucasoid', 'Negroid', 'Mongoloid']
    fopen = open(home + "\\Attributes\\race_list.dat", mode='wb')
    pickle.dump(race_list, fopen)
    fopen.close()


def scenario1():
    home = os.path.dirname(os.getcwd())

    fopen = open(home + "\\Attributes\\train_dataset.dat", mode='rb')
    train = pickle.load(fopen)
    fopen.close()

    import matplotlib.pyplot as plt
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.axis('off')
        plt.imshow(train[i][0], cmap='gray')

    model = cl.CNN_model_1(7)
    cl.train_model(train, model, 0, 4)

    plt.show()


def scenario2():
    home = os.path.dirname(os.getcwd())

    fopen = open(home + "\\Attributes\\train_dataset.dat", mode='rb')
    train = pickle.load(fopen)
    fopen.close()

    train = mt.convert_to_grayscale(train)

    import matplotlib.pyplot as plt
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.axis('off')
        plt.imshow(train[i][0], cmap='gray')

    model = cl.CNN_model_1(3)
    cl.train_model(train, model, 0, 2)

    plt.show()


def scenario5():
    home = os.path.dirname(os.getcwd())

    fopen = open(home + "\\Attributes\\train_dataset.dat", mode='rb')
    train = pickle.load(fopen)
    fopen.close()

    import matplotlib.pyplot as plt
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.axis('off')
        plt.imshow(train[i][0], cmap='gray')

    model = cl.CNN_model_2(3)
    cl.train_model(train, model, 0, 5)

    plt.show()


def scenario6():
    home = os.path.dirname(os.getcwd())

    fopen = open(home + "\\Attributes\\train_dataset.dat", mode='rb')
    train = pickle.load(fopen)
    fopen.close()

    import matplotlib.pyplot as plt
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.axis('off')
        plt.imshow(train[i][0], cmap='gray')

    model = cl.CNN_model_3(3)
    cl.train_model(train, model, 0, 6)

    plt.show()


def scenario7():
    home = os.path.dirname(os.getcwd())

    fopen = open(home + "\\Attributes\\train_dataset.dat", mode='rb')
    train = pickle.load(fopen)
    fopen.close()

    import matplotlib.pyplot as plt
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.axis('off')
        plt.imshow(train[i][0], cmap='gray')

    model = cl.CNN_model_4(3)
    cl.train_model(train, model, 0, 7)

    plt.show()


def scenario8():
    home = os.path.dirname(os.getcwd())

    fopen = open(home + "\\Attributes\\train_dataset.dat", mode='rb')
    train = pickle.load(fopen)
    fopen.close()

    import matplotlib.pyplot as plt
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.axis('off')
        plt.imshow(train[i][0], cmap='gray')

    model = cl.CNN_model_5(3)
    cl.train_model(train, model, 0, 8)

    plt.show()


if __name__ == '__main__':
    scenario7()

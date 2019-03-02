import os, sys
home = os.path.dirname(os.getcwd())
sys.path.insert(0, home + '\\Preprocess')

import load_train_images as lti
import methods as mt
import pickle
import random
import classifier as cl


def create_dataset():
    imgs = lti.load_all_train_images()

    home = os.path.dirname(os.getcwd())
    fopen = open(home + "\\Attributes\\races.dat", mode='rb')
    races = pickle.load(fopen)
    fopen.close()

    dataset = []
    conv_race_list = ['None', 'Caucasoid', 'Negroid', 'Mongoloid', 'Mongoloid', 'Mongoloid', 'Caucasoid']

    for img, race in zip(imgs, races):
        if race != 0:
            dataset.append([img, conv_race_list[race]])

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

    model = cl.CNN_model_1(3)
    cl.train_model(train, model, 0)

    model.save(home + "\\Attributes\\CNN_scenario1.h5")

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
    cl.train_model(train, model, 0)

    model.save(home + "\\Attributes\\CNN_scenario2.h5")

    plt.show()


if __name__ == '__main__':
    scenario1()

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
    conv_race_list = ['None', 'Caucasoid', 'Negroid', 'Mongoloid', 'Mongoloid', 'Australoid', 'Caucasoid']

    for img, race in zip(imgs, races):
        if race != 0:
            dataset.append([img, conv_race_list[race]])

    fopen = open(home + "\\Attributes\\dataset.dat", mode='wb')
    pickle.dump(dataset, fopen)
    fopen.close()


def scenario1():
    home = os.path.dirname(os.getcwd())

    fopen = open(home + "\\Attributes\\dataset.dat", mode='rb')
    dataset = pickle.load(fopen)
    fopen.close()

    random.shuffle(dataset)

    train, test = mt.split_test_train(dataset)

    print(len(train))
    print(len(test))

    import matplotlib.pyplot as plt
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.axis('off')
        plt.imshow(dataset[i][0], cmap='gray')

    model = cl.CNN_model_1(3)
    cl.train_model(train, model, 0)

    plt.show()


if __name__ == '__main__':
    scenario1()

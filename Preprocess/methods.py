from PIL import Image
import numpy as np
import random


def convert_to_grayscale(dataset):
    new_dataset = []

    for data in dataset:
        conv = Image.fromarray(data[0]).convert('L')
        conv = np.array(conv)
        conv = np.array([conv, conv, conv]).reshape((3, 128, 96)).transpose(1, 2, 0)
        new_dataset.append([conv, data[1]])

    return new_dataset


def split_test_train(dataset):
    train = []
    test = []

    for data in dataset:
        ra = random.random()

        if ra < 0.9:
            train.append(data)
        else:
            test.append(data)

    return train, test

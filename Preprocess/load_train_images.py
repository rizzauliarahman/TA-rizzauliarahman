from PIL import Image
import os
import numpy as np
import load_train_races as ltr
import pickle


def load_train_image(filename):
    home = os.path.dirname(os.getcwd())

    img = Image.open(home + "\\Dataset\\Face Images\\" + filename)
    img = img.resize((96, 128))
    arimg = np.array(img)

    return arimg


def load_all_train_images():
    home = os.path.dirname(os.getcwd())

    fp = open(home + "\\Attributes\\imgs.dat", mode='rb')
    imgs_name = pickle.load(fp)

    imgs = []

    for fname in imgs_name:
        img = load_train_image(fname)
        imgs.append(img)

    return imgs

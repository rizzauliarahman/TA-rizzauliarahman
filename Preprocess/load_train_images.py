from PIL import Image
import os
import numpy as np
import load_train_races as ltr

def load_train_image(filename):
    home = os.path.dirname(os.getcwd())

    img = Image.open(home + "\\Dataset\\Face Images\\" + filename)
    img.resize((192, 256))
    img = np.array(img)

    return img



import os, sys
home = os.path.dirname(os.getcwd())
sys.path.insert(0, home + '\\Preprocess')

import load_train_images as lti
import pickle


def main():
    imgs = lti.load_all_train_images()

    home = os.path.dirname(os.getcwd())
    fopen = open(home + "\\Attributes\\races.dat", mode='rb')
    races = pickle.load(fopen)
    fopen.close()

    dataset = []
    conv_race_list = [99, 2, 1, 0, 0, 3, 2]

    for img, race in zip(imgs, races):
        if race != 0:
            dataset.append([img, conv_race_list[race]])

    fopen = open(home + "\\Attributes\\dataset.dat", mode='wb')
    pickle.dump(dataset, fopen)
    fopen.close()

    # import matplotlib.pyplot as plt
    # for img in imgs[:9]:
    #     plt.subplot(3, 3, imgs.index(img)+1)
    #     plt.axis('off')
    #     plt.imshow(img)
    #
    # plt.show()


if __name__ == '__main__':
    main()

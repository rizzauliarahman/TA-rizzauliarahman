import os, sys
home = os.path.dirname(os.getcwd())
sys.path.insert(0, home + '\\Preprocess')

import load_train_images as lti


def main():
    img = lti.load_train_image("Aaron_Boothe_5_oval.jpg")

    print("Done")

    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    main()

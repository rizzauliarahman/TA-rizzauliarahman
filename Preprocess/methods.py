from PIL import Image
import numpy as np
import random
import tabulate as tb


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


def count_accuracy(classes, y_test):
    right = sum([1 if np.argmax(classes[i]) == y_test[i] else 0 for i in range(len(classes))])
    print(right)
    print(len(classes))

    return right / len(classes)


def confusion_matrix(classes, y_test, race_list, txtopen):
    txtopen.write("============ CONFUSION MATRIX ===============\n")
    for i in range(len(race_list)):
        txtopen.write("=========================================\n")
        txtopen.write("Class : " + str(race_list[i]).upper() + "\n\n")

        Tp = sum([1 if np.argmax(classes[j]) == i and y_test[j] == i else 0 for j in range(len(classes))])
        Fp = sum([1 if np.argmax(classes[j]) == i and y_test[j] != i else 0 for j in range(len(classes))])
        Fn = sum([1 if np.argmax(classes[j]) != i and y_test[j] == i else 0 for j in range(len(classes))])
        Tn = sum([1 if np.argmax(classes[j]) != i and y_test[j] != i else 0 for j in range(len(classes))])

        Precision = Tp / (Tp + Fp)
        Recall = Tp / (Tp + Fn)

        f1 = (2 * Tp) / ((2 * Tp) + Fp + Fn)
        accu = (Tp + Tn) / (Tp + Tn + Fp + Fn)

        s = tb.tabulate([['Predicted T', Tp, Fp], ['Predicted F', Fn, Tn]], headers=['', 'Actual T', 'Actual F'],
                        tablefmt='orgtbl')
        txtopen.write(s)
        txtopen.write("\n")
        txtopen.write("\nPrecision: %.3f\n" % Precision)
        txtopen.write("Recall: %.3f\n" % Recall)
        txtopen.write("F1-Score: %.3f\n" % f1)
        txtopen.write("Accuracy: %.3f%%\n\n" % (accu * 100))

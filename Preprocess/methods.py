from PIL import Image
import numpy as np
import random
import tabulate as tb


def convert_to_grayscale(dataset):
    new_dataset = []

    for data in dataset:
        conv = Image.fromarray(data[0]).convert('L')
        conv = np.array(conv)
        conv = np.array([conv, conv, conv]).reshape((3, 96, 96)).transpose(1, 2, 0)
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

    return right / len(classes)


def confusion_matrix(classes, y_test, race_list, txtopen):
    txtopen.write("====================== CONFUSION MATRIX =========================\n\n")

    total_Precision = 0.0
    total_Recall = 0.0
    total_F1 = 0.0

    tables = []

    for i in range(len(race_list)):
        Tp = sum([1 if np.argmax(classes[j]) == i and y_test[j] == i else 0 for j in range(len(classes))])
        Fp = sum([1 if np.argmax(classes[j]) == i and y_test[j] != i else 0 for j in range(len(classes))])
        Fn = sum([1 if np.argmax(classes[j]) != i and y_test[j] == i else 0 for j in range(len(classes))])
        Tn = sum([1 if np.argmax(classes[j]) != i and y_test[j] != i else 0 for j in range(len(classes))])

        tbl = []
        for k in range(len(race_list)):
            tbl.append(sum([1 if np.argmax(classes[j]) == k and y_test[j] == i else 0 for j in range(len(classes))]))

        tables.append(tbl)

        total_Precision += Tp / (Tp + Fp)
        total_Recall += Tp / (Tp + Fn)

        total_F1 += (2 * Tp) / ((2 * Tp) + Fp + Fn)

    header = ['']
    header.extend([("Actual\n%s" % s) for s in race_list])

    datas = []
    for i in range(len(race_list)):
        data = ["Predicted %s" % race_list[i]]
        data.extend([tables[j][i] for j in range(len(race_list))])
        datas.append(data)

    s = tb.tabulate(datas, headers=header, tablefmt='orgtbl', numalign="center")
    txtopen.write(s)
    txtopen.write("\n")
    txtopen.write("\nPrecision: %.3f\n" % (total_Precision / len(race_list)))
    txtopen.write("Recall: %.3f\n" % (total_Recall / len(race_list)))
    txtopen.write("F1-Score: %.3f\n" % (total_F1 / len(race_list)))

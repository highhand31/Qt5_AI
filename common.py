# import tensorflow as tf
# from tensorflow.contrib import slim
import cv2
import os
import numpy as np
import matplotlib.image as mpimg
import time
import matplotlib.pyplot as plt


def data_load(pic_path, train_ratio, resize=None, shuffle=True, normalize=True, has_dir=True):
    labels = {}
    x_train = []
    x_train_label = []
    x_test = []
    x_test_label = []


    if resize is not None:
        width = resize[0]
        height = resize[1]

        print('resize width = ', width)
        print('resize height = ', height)

    if train_ratio > 1:
        train_ratio = 1

    if has_dir is True:
        for num, dirs in enumerate(os.scandir(pic_path)):

            if dirs.is_dir():
                #print(dirs.name)
                labels[dirs.name] = num
                file_path = os.path.join(pic_path, dirs.name)
                #print(file_path)
                files = [file.path for file in os.scandir(file_path) if file.is_file()]
                print("Picture number of dir({}) is {} ".format(file_path, len(files)))
                pic_length = len(files)

                train_num = int(pic_length * train_ratio)
                test_num = pic_length - train_num

                print('train data number = ', train_num)
                print('test data number = ', test_num)

                for pic_num, file in enumerate(files):

                    # img = cv2.imread(file)
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = mpimg.imread((file))

                    # 圖片進行resize
                    if resize is not None:
                        img = cv2.resize(img, (width, height))
                    if pic_num < train_num:
                        x_train.append(img)
                        x_train_label.append(labels[dirs.name])
                    else:
                        x_test.append(img)
                        x_test_label.append(labels[dirs.name])

    else:#資料夾內沒有資料夾，只有照片

        files = [file.path for file in os.scandir(pic_path) if file.is_file()]
        print("Picture number of dir({}) is {} ".format(pic_path, len(files)))
        pic_length = len(files)
        train_num = int(pic_length * train_ratio)
        test_num = pic_length - train_num
        print('train data number = ', train_num)
        print('test data number = ', test_num)

        for pic_num,file in enumerate(files):
            #print(file)
            # img = cv2.imread(file)
            #print(img.shape)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = mpimg.imread((file))

            # 圖片進行resize
            if resize is not None:
                img = cv2.resize(img, (width, height))
            if pic_num < train_num:
                x_train.append(img)
                x_train_label.append(0)
            else:
                x_test.append(img)
                x_test_label.append(0)
            # flatten_num = img.shape[0]*img.shape[1]*img.shape[2]
            # img = img.reshape(flatten_num)

    #-------------------
    x_train = np.array(x_train)
    x_train_label = np.array(x_train_label)
    x_test = np.array(x_test)
    x_test_label = np.array(x_test_label)

    # 將資料進行shuffle
    if shuffle is True:
        indice = np.random.permutation(x_train_label.shape[0])
        x_train = x_train[indice]
        x_train_label = x_train_label[indice]

        indice = np.random.permutation(x_test_label.shape[0])
        x_test = x_test[indice]
        x_test_label = x_test_label[indice]
        print("The data shuffle is done")

    if normalize is True:
        x_train = x_train.astype("float32")
        x_train = x_train / 255

        x_test = x_test.astype("float32")
        x_test = x_test / 255
        print("The data normalization is done")

    print(labels)

    return (x_train, x_train_label, x_test, x_test_label)

if __name__ == "__main__":
    pic_path = r'./xxx'
    (x_train_2, x_train_label_2, x_test_2, x_test_label_2) = data_load(pic_path, 0.5, resize=(64, 64),
                                                                          shuffle=True, normalize=True)
    # print('x_train shape = ',x_train_2.shape)
    # print('x_train_label shape = ',x_train_label_2.shape)
    print('x_test shape = ', x_test_2.shape)
    print('x_test_label shape = ', x_test_label_2.shape)
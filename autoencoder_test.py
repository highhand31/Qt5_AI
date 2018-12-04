import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.contrib import slim
import cv2
import os
import numpy as np
import socket as sc
import time
import matplotlib.pyplot as plt
import common as cm
import json


save_path = r"model_saver\AE_circle"
out_dir_prefix=os.path.join(save_path,"model")
height = 64
width = 64
epochs = 50
GPU_ratio = 0.5
batch_size = 12
train_ratio = 0.7

class AE():

    def __init__(self, input_dim=[None, 64, 64, 3], save_path="model_saver\ckpt"):

        self.input_dim = input_dim
        self.save_path = save_path
        self.kel_x = 9
        self.kel_y = 9
        self.deep = 64
        self.UDP_init_flag = False
        self.UDP_init()
        #self.data_init()
        self.__build_model()
        #self.train(self.x_train,self.x_test_2,self.x_test_label_2, epochs=10, fine_tune=False)

        #print("initial")

    def data_init(self):
        save_path = r"model_saver\AE_circle"
        out_dir_prefix = os.path.join(save_path, "model")
        height = 64
        width = 64
        epochs = 50
        GPU_ratio = 0.5
        batch_size = 12
        train_ratio = 0.7

        pic_path = r'./goodsamples'
        (self.x_train, self.x_train_label, no1, no2) = cm.data_load(pic_path, train_ratio=1,
                                                                    resize=(width, height),
                                                                    has_dir=False)
        print(self.x_train.shape)
        # self.ui.plainTextEdit.appendPlainText('x_train.shape = {}'.format(x_train.shape))
        print('x_train.shape = {}'.format(self.x_train.shape))
        # self.ui.label.setText('x_train.shape = {}'.format(x_train.shape))

        # prepare test data
        pic_path = r'./xxx'
        (self.x_train_2, self.x_train_label_2, self.x_test_2, self.x_test_label_2) = cm.data_load(pic_path, train_ratio,
                                                                              resize=(width, height), shuffle=True,
                                                                              normalize=True)

        print('x_test shape = ', self.x_test_2.shape)
        # self.ui.plainTextEdit.appendPlainText('x_test shape = {}'.format(x_test_2.shape))
        print('x_test shape = {}'.format(self.x_test_2.shape))

        print('x_test_label shape = ', self.x_test_label_2.shape)
        # self.ui.plainTextEdit.appendPlainText('xx_test_label shape = {}'.format(x_test_label_2.shape))
        print('x_test_label shape = {}'.format(self.x_test_label_2.shape))


    def UDP_init(self):
        self.sock = sc.socket(sc.AF_INET, sc.SOCK_DGRAM)
        self.send_address = ("127.0.0.1",5002)
        #json format {"mode":str,"msg":str, "train loss":list, "acc":list}
        #if UDP transmit msg -->"mode":"msg"
        #if UDP transmit training data -->"mode":"dict"
        send_msg = {"mode":"msg","msg":"AI Model UDP TX init ok"}
        send_msg = json.dumps(send_msg)
        self.UDP_send(send_msg,self.send_address)
        self.UDP_dict = {}
        self.UDP_init_flag = True

    def UDP_send(self,send_msg,send_address):

        send_msg = bytes(send_msg, encoding="gbk")
        self.sock.sendto(send_msg, send_address)

    def __build_model(self):

        self.input_x = tf.placeholder(tf.float32, self.input_dim, name="input_x")

        self.prediction = self.__inference(self.input_x)

        self.save_path = save_path

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.saver = tf.train.Saver(max_to_keep=20)

        self.out_dir_prefix = os.path.join(save_path, "model")

        self.loss = tf.reduce_mean(tf.pow(self.prediction - self.input_x, 2),name="output")

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

    def __inference(self, input_x):

        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.conv2d_transpose],
                            activation_fn=tf.nn.relu,
                            # normalizer_fn=slim.batch_norm,
                            # normalizer_params={'is_training': True, 'decay': 0.995},
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            reuse=tf.AUTO_REUSE):
            self.net = slim.conv2d(self.input_x, 64, [self.kel_x, self.kel_y], scope='encode1')
            self.net = slim.max_pool2d(self.net, [2, 2], scope='pool1')
            print("encode1 shape = ", self.net.shape)

            # net shape = 32 x 32 x 64

            self.net = slim.conv2d(self.net, 64, [self.kel_x, self.kel_y], scope='encode2')
            self.net = slim.max_pool2d(self.net, [2, 2], scope='pool2')
            print("encode2 shape = ", self.net.shape)

            # net shape = 16 x 16 x 64

            self.net = slim.conv2d(self.net, 64, [self.kel_x, self.kel_y], scope='encode3')
            self.net = slim.max_pool2d(self.net, [2, 2], scope='pool3')
            print("encode3 shape = ", self.net.shape)

            # net shape = 8 x 8 x 64

            self.net = slim.conv2d(self.net, 64, [self.kel_x, self.kel_y], scope='encode4')
            self.net = slim.max_pool2d(self.net, [2, 2], scope='pool4')
            print("encode4 shape = ", self.net.shape)

            # net shape = 4 x 4 x 64

            self.net = slim.conv2d_transpose(self.net, 64, [2, 2], stride=2, activation_fn=None, padding="VALID",
                                             scope="unpool1")
            self.net = slim.conv2d(self.net, 64, [self.kel_x, self.kel_y], scope='decode1_1')
            print("decode1 shape = ", self.net.shape)

            # net shape = 8 x 8 x 64

            self.net = slim.conv2d_transpose(self.net, 64, [2, 2], stride=2, activation_fn=None, padding="VALID",
                                             scope="unpool2")
            self.net = slim.conv2d(self.net, 64, [self.kel_x, self.kel_y], scope='decode2_3')
            print("decode2 shape = ", self.net.shape)

            # net shape = 16 x 16 x 64

            self.net = slim.conv2d_transpose(self.net, 64, [2, 2], stride=2, activation_fn=None, padding="VALID",
                                             scope="unpool3")
            # net = unpooling(net,[64,64])
            self.net = slim.conv2d(self.net, 64, [self.kel_x, self.kel_y], scope='decode3_1')
            print("decode3 shape = ", self.net.shape)

            # net shape = 32 x 32 x 64

            self.net = slim.conv2d_transpose(self.net, 64, [2, 2], stride=2, activation_fn=None, padding="VALID",
                                             scope="unpool4")
            # net = unpooling(net,[64,64])
            self.net = slim.conv2d(self.net, 3, [self.kel_x, self.kel_y], scope='decode4')#name = "decode4/Relu"
            # net +=net1
            print("decode4 shape = ", self.net.shape)

            # net shape = 64 x 64 x 3

            return self.net

    def train(self, train_data, test_data, test_label, GPU_ratio=0.2, epochs=50, batch_size=16, fine_tune=False,
              save_ckpt=True):

        #             self.GPU_ratio = GPU_ratio
        #             self.epochs = epochs
        #             self.batch_size = batch_size

        #var init
        self.train_loss_set = []
        self.acc_set = []

        # 計算total batch
        total_batches = train_data.shape[0] // batch_size
        if train_data.shape[0] % batch_size:
            total_batches += 1

        # 設定GPU參數
        config = tf.ConfigProto(log_device_placement=True,
                                allow_soft_placement=True,  # 允許當找不到設備時自動轉換成有支援的設備
                                )
        config.gpu_options.per_process_gpu_memory_fraction = GPU_ratio

        with tf.Session(config=config) as sess:

            if fine_tune is True:  # 使用已經訓練好的權重繼續訓練
                files = [file.path for file in os.scandir(self.save_path) if file.is_file()]

                if not files:  # 沒有任何之前的權重

                    sess.run(tf.global_variables_initializer())

                    print('no previous model param can be used')

                else:

                    self.saver.restore(sess, tf.train.latest_checkpoint(self.save_path))

                    print('use previous model param')

            else:
                sess.run(tf.global_variables_initializer())
                print('no previous model param can be used')

            for epoch in range(epochs):

                for index in range(total_batches):

                    num_start = index * batch_size
                    num_end = num_start + batch_size

                    if num_end >= train_data.shape[0] and batch_size > 1:
                        num_end = train_data.shape[0] - 1

                    sess.run(self.optimizer, feed_dict={self.input_x: train_data[num_start:num_end]})

                # compute mean loss after a epoch
                train_loss = []
                for index in range(train_data.shape[0]):
                    single_loss = sess.run(self.loss, feed_dict={self.input_x: train_data[index:index + 1]})

                    train_loss.append(single_loss)

                train_loss = np.array(train_loss)
                train_loss = np.mean(train_loss)

                test_loss = []
                acc = 0
                for index in range(test_data.shape[0]):

                    single_loss = sess.run(self.loss, feed_dict={self.input_x: test_data[index:index + 1]})

                    test_loss.append(single_loss)

                    if single_loss > train_loss:

                        if test_label[index] == 0 or test_label[index] == 1:
                            acc += 1

                    elif single_loss >= 0:

                        if test_label[index] == 2:
                            acc += 1

                acc /= test_data.shape[0]
                test_loss = np.array(test_loss)
                test_loss = np.mean(test_loss)
                msg = "Epoch {}\ntrain set loss = {}".format(epoch, train_loss)
                print(msg)
                #self.UDP_send(msg,self.send_address)

                msg = "test set loss = {}, accuracy = {}".format(test_loss, acc)
                print(msg)
                #self.UDP_send(msg, self.send_address)

                # 紀錄資料:本次epoch ckpt檔
                if save_ckpt is True:
                    model_save_path = self.saver.save(sess, self.out_dir_prefix, global_step=epoch)

                    print('Save model checkpoint to ', model_save_path)

                graph = tf.get_default_graph().as_graph_def()
                # output_graph_def = graph_util.convert_variables_to_constants(sess, graph,['output'])  # graph也可以直接填入sess.graph_def
                output_graph_def = graph_util.convert_variables_to_constants(sess, graph,
                                                                             ['decode4/Relu'])  # graph也可以直接填入sess.graph_def


                # 'model_saver/'為置放的資料夾，'combined_model.pb'為檔名
                with tf.gfile.GFile("model_saver/pb_test_model.pb", "wb") as f:

                    f.write(output_graph_def.SerializeToString())

                #training data collection
                # 原本train_loss的型態是numpy.float32，進行json序列化會失敗，要改成python內建的float型態才能進行json序列化

                self.train_loss_set.append(float(train_loss))
                # print("type of train loss = {}".format(type(train_loss)))
                # print("type of acc = {}".format(type(acc)))
                self.acc_set.append(acc)

                # self.UDP_dict["train loss"] = self.train_loss_set
                # self.UDP_dict["acc"] = self.acc_set
                # for name,value in self.UDP_dict.items():
                #     print("{} = {}".format(name,value))

                #UDP sender
                if self.UDP_init_flag:
                    self.UDP_dict["mode"] = "dict"
                    self.UDP_dict["train loss"] = self.train_loss_set
                    self.UDP_dict["acc"] = self.acc_set
                    jsonObj = json.dumps(self.UDP_dict)#若要傳送dict，需要先進行json化
                    # jsonObj = json.dumps(self.train_loss_set)
                    # self.UDP_send("UDP sender:loss = {},acc = {}".format(train_loss,acc),self.send_address)
                    self.UDP_send(jsonObj,self.send_address)

                #return train_loss, acc

    def eval(self, train_data, test_data, test_label, GPU_ratio=0.2):

        total_loss = []
        single_loss = 0

        # 設定GPU參數
        config = tf.ConfigProto(log_device_placement=True,
                                allow_soft_placement=True,  # 允許當找不到設備時自動轉換成有支援的設備
                                )
        config.gpu_options.per_process_gpu_memory_fraction = GPU_ratio

        with tf.Session(config=config) as sess:

            self.saver.restore(sess, tf.train.latest_checkpoint(self.save_path))

            print('use previous model param')

            for i in range(train_data.shape[0]):
                single_loss = sess.run(self.loss, feed_dict={self.input_x: train_data[i:i + 1]})

                total_loss.append(single_loss)

            total_loss = np.array(total_loss)

            train_ave_loss = np.mean(total_loss)

            total_loss = []

            acc = 0

            for i in range(test_data.shape[0]):

                single_loss = sess.run(self.loss, feed_dict={self.input_x: test_data[i:i + 1]})

                if single_loss > train_ave_loss:

                    if test_label[i] == 0 or test_label[i] == 1:
                        acc += 1

                elif single_loss >= 0:

                    if test_label[i] == 2:
                        acc += 1

                total_loss.append(single_loss)

            acc /= test_data.shape[0]

            total_loss = np.array(total_loss)

            test_ave_loss = np.mean(total_loss)

            return train_ave_loss, test_ave_loss, acc

if __name__ == "__main__":

    #prepare training data
    pic_path = r'./goodsamples'
    #pic_path = r"G:/我的雲端硬碟/Python/Code/Pycharm/P_UI_Test/goodsamples"
    # pic_path = r"G://我的雲端硬碟//Python//Code//Pycharm//P_UI_Test//goodsamples"
    #pic_path = "G:\\我的雲端硬碟\\Python\\Code\\Pycharm\\P_UI_Test\\goodsamples"
    # pic_path = 'C:\Users\highh\OneDrive\Desktop\goodsamples'
    #pic_path = "E:/dataset/Surface_detection/0"
    (x_train, x_train_label, no1, no2) = cm.data_load(pic_path, train_ratio=1, resize=(width, height), has_dir=False)
    print(x_train.shape)
    # print(x_train_label)


    #prepare test data
    pic_path = r'./xxx'
    (x_train_2,x_train_label_2,x_test_2,x_test_label_2) = cm.data_load(pic_path,train_ratio,resize=(width,height),shuffle = True,normalize = True)
    # print('x_train shape = ',x_train_2.shape)
    # print('x_train_label shape = ',x_train_label_2.shape)
    print('x_test shape = ',x_test_2.shape)
    print('x_test_label shape = ',x_test_label_2.shape)


    ae = AE()
    ae.train(x_train,x_test_2,x_test_label_2,epochs = 5)
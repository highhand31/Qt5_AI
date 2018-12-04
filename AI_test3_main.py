import sys
#from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication,QFileDialog,QMainWindow
from PyQt5.QtCore import QObject,pyqtSignal,QThread,Qt
#import PyQt5.sip #fro Pyinstaller
# from MatplotlibWidget import MatplotlibWidget
# import time
from AI_test3 import Ui_Form
import common as cm
from queue import Queue
import autoencoder_test as auto
import socket as sc
# import numpy as np
import json

# The new Stream Object which replaces the default stream associated with sys.stdout
# This object just puts data in a queue!
class WriteStream(object):
    def __init__(self,queue):
        self.queue = queue

    def write(self, text):
        self.queue.put(text)

class UDP_listener(QObject):
    UDP_signal = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.sock = sc.socket(sc.AF_INET, sc.SOCK_DGRAM)
        address = ("127.0.0.1", 5002)
        self.sock.bind(address)
        print("UDP listener init ok")

    def run(self):
        while True:
            data, addr = self.sock.recvfrom(1024)
            data = str(data, encoding="gbk")
            data = json.loads(data)
            # for name,value in data.items():
            #     print("{} = {}".format(name,value))
            #print("I got data")
            #print(data, addr)
            self.UDP_signal.emit(data)

class MyReceiver(QObject):
    mysignal = pyqtSignal(str)

    def __init__(self,queue,*args,**kwargs):
        QObject.__init__(self,*args,**kwargs)
        self.queue = queue

    #@pyqtSlot()
    def run(self):
        while True:
            text = self.queue.get()
            self.mysignal.emit(text)


class AI_model(QObject):

    finish_flag = pyqtSignal()
    def __init__(self,AI_var):
        super().__init__()
        self.AI_var = AI_var
        ''' 
        dictionary description
            train data param:
                train_dir_name,resize_height,resize_width,train_ratio,train_has_dir
            test data param:
                test_dir_name,test_ratio,test_has_dir,
            training param:
                epoch,batch_size,GPU_ratio,fine_tune,save_ckpt
        '''
        #train data param:
        self.train_dir_name = self.AI_var["train_dir_name"]
        self.resize_height = self.AI_var["resize_height"]
        self.resize_width = self.AI_var["resize_width"]
        self.train_ratio = 1.0
        self.train_has_dir = False
        #test data param
        self.test_dir_name = self.AI_var['test_dir_name']
        self.test_ratio = 0
        self.test_has_dir = True
        #training param
        self.epoch = self.AI_var['epoch']
        self.batch_size = self.AI_var['batch_size']
        self.GPU_ratio = self.AI_var['GPU_ratio']
        self.fine_tune = self.AI_var['fine_tune']
        self.save_ckpt = self.AI_var['save_ckpt']

        print("AI training params from UI inputs are shown below:")
        for name,value in self.AI_var.items():
            print("{} = {}".format(name,value))

    def run(self):
        print("Start train data pre-process\n")
        (x_train, y_train_label, no1, no2) = cm.data_load(self.train_dir_name, self.train_ratio,
                                                          resize=(self.resize_width,self.resize_height),
                                                          has_dir=self.train_has_dir)
        print('Training data shape = {}'.format(x_train.shape))


        print("Start test data pre-process\n")
        (x_train_2, y_train_label_2, x_test, y_test_label) = cm.data_load(self.test_dir_name, self.test_ratio,
                                                          resize=(self.resize_width, self.resize_height),
                                                          has_dir=self.test_has_dir)
        print('Test data shape = {}'.format(x_test.shape))
        print('Test label shape = {}'.format(y_test_label.shape))

        self.ae = auto.AE()
        self.ae.train(x_train,x_test,y_test_label,self.GPU_ratio,self.epoch,
                      self.batch_size,self.fine_tune,self.save_ckpt)

        self.finish_flag.emit()


class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        #variables init
        self.train_dir_name = self.ui.train_dir_display.placeholderText()
        self.test_dir_name = self.ui.test_dir_display_2.placeholderText()
        self.AI_var = {}
        self.epoch = self.ui.epochEdit.text()
        self.batch_size = self.ui.batchEdit.text()
        self.GPU_ratio = self.ui.gpuratioEdit.text()

        #sys.stdout change to queue
        self.queue = Queue()
        self.stdout_default = sys.stdout
        sys.stdout = WriteStream(self.queue)  # 將stdout轉接至queue，之後的print()內容都會丟到queue裡

        # thread2: receive stdout via queue
        self.thread_2 = QThread()
        self.my_receiver = MyReceiver(self.queue)
        self.my_receiver.mysignal.connect(self.append_text)
        self.my_receiver.moveToThread(self.thread_2)
        self.thread_2.started.connect(self.my_receiver.run)
        self.thread_2.start()

        #thread3:UDP listener
        self.thread_UDP_listener = QThread()
        self.UDP_listen = UDP_listener()
        self.UDP_listen.UDP_signal.connect(self.UDP_get_data)
        self.UDP_listen.moveToThread(self.thread_UDP_listener)
        self.thread_UDP_listener.started.connect(self.UDP_listen.run)
        self.thread_UDP_listener.start()


        #setup events
        self.ui.btn_train_dir.clicked.connect(self.btn_train_dir_clicked)
        self.ui.btn_test_dir.clicked.connect(self.btn_test_dir_clicked)
        self.ui.btn_data_process.clicked.connect(self.btn_AI_training_clicked)


        #UI plot code(temp)
        # add by Johnny
        # self.matplotlibwidget_dynamic = MatplotlibWidget(self.tab)
        # self.matplotlibwidget_dynamic.setEnabled(True)
        # self.matplotlibwidget_dynamic.setHidden(True)
        # self.matplotlibwidget_dynamic.setGeometry(QtCore.QRect(0, 500, 620, 450))
        # self.matplotlibwidget_dynamic.setObjectName("matplotlibwidget_dynamic")



    def btn_train_dir_clicked(self):
        self.train_dir_name = QFileDialog.getExistingDirectory(self, "select Train data directory")
        self.train_dir_name = repr(self.train_dir_name)[1:-1]
        self.ui.train_dir_display.setText(self.train_dir_name)

    def btn_test_dir_clicked(self):
        self.test_dir_name = QFileDialog.getExistingDirectory(self, "select Test data directory")
        self.test_dir_name = repr(self.test_dir_name)[1:-1]
        self.ui.test_dir_display_2.setText(self.test_dir_name)

    def UDP_get_data(self,text):

        try:
            mode = text["mode"]
            if mode == "msg":
                print(text["msg"])
            elif mode == "dict":
                # for name,value in text.items():
                #     print("{} = {}".format(name,value))

                # train_loss = text["train loss"]
                # train_loss = np.array(train_loss)
                #self.fig.clear(True)
                self.fig.suptitle('Loss and Accuracy')
                #l = [np.random.randint(0, 10) for i in range(4)]
                #self.axes.plot([0, 1, 2, 3], l, 'r')
                #去除上一次的數值
                self.axes.cla()
                #繪製2條線
                self.axes.plot(text["train loss"],label="train loss")
                self.axes.plot(text["acc"],label="accuracy")
                #設置圖例位置
                self.axes.legend(loc="best", shadow=True)
                self.axes.set_ylabel('loss')
                self.axes.set_xlabel('epoch')
                self.axes.grid(True)
                self.ui.matplotlibwidget_dynamic.mpl.draw()

        except ValueError:
            print("UDP receives messages with error format")
        #print(text["train loss"])
        # for name,value in text.items():
        #     print("{} = {}".format(name.value))
        # for name,value in text.items():
        #     print("{} is {}".format(name,value))
        # if not text:

        #     self.fig.suptitle('plot test')
        #     # t = np.arange(0.0, 3.0, 0.01)
        #     # s = np.sin(2 * np.pi * t)
        #     # self.axes.plot(t, s)


    def append_text(self,text):
        self.ui.consoleEdit.append(text)

    def btn_AI_training_clicked(self):
        # self.ui.consoleEdit.setGeometry(QtCore.QRect(0, 440, 385, 301))#left, top, width and height

        #set plot visible
        self.ui.matplotlibwidget_dynamic.setHidden(False)
        #self.ui.matplotlibwidget_dynamic.mpl.start_static_plot()
        self.fig = self.ui.matplotlibwidget_dynamic.mpl.fig
        self.axes = self.ui.matplotlibwidget_dynamic.mpl.axes
        # self.fig.suptitle('plot test')
        # t = np.arange(0.0, 3.0, 0.01)
        # s = np.sin(2 * np.pi * t)
        # self.axes.plot(t, s)
        # self.axes.set_ylabel('Loss')
        # self.axes.set_xlabel('Epoch')
        # self.axes.grid(True)



        #set btn disabled
        self.ui.btn_data_process.setEnabled(False)

        #clear textEdit
        self.ui.consoleEdit.clear()

        #check all inputs are right
        self.AI_param_check_flag = self.AI_param_check()

        if self.AI_param_check_flag:
            self.ui.consoleEdit.append("All UI inputs checked ok, start connect AI_model\n")
            # setup threads
            #thread1:execute data process and AI model
            self.thread = QThread()
            self.AI_model = AI_model(self.AI_var)
            self.AI_model.finish_flag.connect(self.AI_model_finish_process)
            self.AI_model.moveToThread(self.thread)  # 將要執行的方法放到線程裡
            self.thread.started.connect(self.AI_model.run)  # 設定線程執行時要執行物件的哪個方法
            self.thread.start()  # 開始執行線程

            # thread2: receive stdout via queue
            # self.thread_2 = QThread()
            # self.my_receiver = MyReceiver(self.queue)
            # self.my_receiver.mysignal.connect(self.append_text)
            # self.my_receiver.moveToThread(self.thread_2)
            # self.thread_2.started.connect(self.my_receiver.run)
            # self.thread_2.start()
        else:
            self.ui.consoleEdit.append("UI inputs are not completed, plz see msg above\n")
            self.ui.btn_data_process.setEnabled(True)


    def AI_param_check(self):
        all_data_checked = True
        self.ui.consoleEdit.append("Start to exam UI inputs")

        # train dir check
        if self.train_dir_name == "":
            self.ui.consoleEdit.setTextColor(Qt.red)
            self.ui.consoleEdit.append("No train dir")
            self.ui.consoleEdit.setTextColor(Qt.black)
            all_data_checked = False
        else:
            self.AI_var["train_dir_name"] = self.train_dir_name

        # resize height check(integer,>0)
        try:#使用try and except檢驗是否為integer
            self.resize_height = int(self.ui.heightEdit.text())
            if self.resize_height > 0:
                self.ui.consoleEdit.append("Resize height value is {}".format(self.resize_height))
                self.ui.consoleEdit.append("Resize height value is checked ok")

                self.AI_var["resize_height"] = self.resize_height

        except ValueError:
            self.ui.consoleEdit.setTextColor(Qt.red)#將以下要顯示的字體顏色(錯誤訊息)變成紅色
            self.ui.consoleEdit.append("Resize height value is not an integer")
            self.ui.consoleEdit.setTextColor(Qt.black)#將顯示的字體顏色更改回黑色
            all_data_checked = False

        # if self.ui.heightEdit.text().isdigit():  # 確認輸入的值是否只有整數數字，不是浮點數
        #     self.resize_height = int(self.ui.heightEdit.text())
        #     if self.resize_height > 0:
        #         self.ui.consoleEdit.append("Resize height value is {}".format(self.resize_height))
        #         self.ui.consoleEdit.append("Resize height value is checked ok")
        #
        #         self.AI_var["resize_height"] = self.resize_height
        # else:
        #     self.ui.consoleEdit.append("Resize height value is not an integer")
        #     all_data_checked = False

        # resize width check(integer,>0)
        if self.ui.widthEdit.text().isdigit():  # 確認輸入的值是否只有整數數字，不是浮點數
            self.resize_width = int(self.ui.widthEdit.text())
            if self.resize_width > 0:
                self.ui.consoleEdit.append("Resize width value is {}".format(self.resize_width))
                self.ui.consoleEdit.append("Resize width value is checked ok")
                self.AI_var["resize_width"] = self.resize_width
        else:
            self.ui.consoleEdit.setTextColor(Qt.red)  # 將以下要顯示的字體顏色(錯誤訊息)變成紅色
            self.ui.consoleEdit.append("Resize width value is not an integer")
            self.ui.consoleEdit.setTextColor(Qt.black)
            all_data_checked = False

        # train ratio(floating number) check
        # try:
        #     self.train_ratio = float(self.ui.train_ratioEdit.text())
        #     if self.train_ratio <= 1.0:
        #         self.ui.consoleEdit.append("train_ratio value is {}".format(self.train_ratio))
        #         self.ui.consoleEdit.append("train_ratio value is checked ok")
        #         self.AI_var["train_ratio"] = self.train_ratio
        #     else:
        #         self.ui.consoleEdit.append("train_ratio value is over 1, must be under 1")
        #         all_data_checked = False
        # except ValueError:
        #     self.ui.consoleEdit.append("train_ratio value is not numeric or float")

        # train data_has dir check
        # self.train_has_dir = False
        # if self.ui.chkB_train_has_dir.isChecked():
        #     self.train_has_dir = True
        #
        # self.AI_var["train_has_dir"] = self.train_has_dir
        # self.ui.consoleEdit.append(str(self.train_has_dir))

        # test dir check
        if self.test_dir_name == "":
            self.ui.consoleEdit.setTextColor(Qt.red)
            self.ui.consoleEdit.append("No test dir")
            self.ui.consoleEdit.setTextColor(Qt.black)
            all_data_checked = False
        else:
            self.AI_var["test_dir_name"] = self.test_dir_name

        # test ratio(floating number) check
        # try:
        #     self.test_ratio = float(self.ui.test_ratioEdit.text())
        #     if self.test_ratio <= 1.0:
        #         self.ui.consoleEdit.append("test_ratio value is {}".format(self.test_ratio))
        #         self.ui.consoleEdit.append("test_ratio value is checked ok")
        #         self.AI_var["test_ratio"] = self.test_ratio
        #     else:
        #         self.ui.consoleEdit.append("test_ratio value is over 1, must be under 1")
        #         all_data_checked = False
        # except ValueError:
        #     self.ui.consoleEdit.append("test_ratio value is not numeric or float")

        # test data_has dir check
        # self.test_has_dir = False
        # if self.ui.chkB_test_has_dir.isChecked():
        #     self.test_has_dir = True
        #
        # self.AI_var['test_has_dir'] = self.test_has_dir
        # self.ui.consoleEdit.append(str(self.test_has_dir))

        # epoch check(integer,must be > 0)
        if self.ui.epochEdit.text().isdigit():  # 確認輸入的值是否只有整數數字，不是浮點數
            self.epoch = int(self.ui.epochEdit.text())
            if self.epoch > 0:
                self.ui.consoleEdit.append("epoch value is {}".format(self.epoch))
                self.ui.consoleEdit.append("epoch value is checked ok")
                self.AI_var["epoch"] = self.epoch
        else:
            self.ui.consoleEdit.setTextColor(Qt.red)
            self.ui.consoleEdit.append("epoch value is not an integer")
            self.ui.consoleEdit.setTextColor(Qt.black)
            all_data_checked = False

        # batch size check(integer,must be > 0)
        if self.ui.batchEdit.text().isdigit():  # 確認輸入的值是否只有整數數字，不是浮點數
            self.batch_size = int(self.ui.batchEdit.text())
            if self.batch_size > 0:
                self.ui.consoleEdit.append("batch_size value is {}".format(self.batch_size))
                self.ui.consoleEdit.append("batch_size value is checked ok")
                self.AI_var["batch_size"] = self.batch_size
        else:
            self.ui.consoleEdit.setTextColor(Qt.red)
            self.ui.consoleEdit.append("batch_size value is not an integer")
            self.ui.consoleEdit.setTextColor(Qt.black)
            all_data_checked = False

        # GPU ratio check(float,>0,<1)
        try:
            self.GPU_ratio = float(self.ui.gpuratioEdit.text())
            if self.GPU_ratio > 0 and self.GPU_ratio <= 1.0:
                self.ui.consoleEdit.append("GPU ratio value is {}".format(self.GPU_ratio))
                self.ui.consoleEdit.append("GPU ratio value is checked ok")
                self.AI_var["GPU_ratio"] = self.GPU_ratio
            else:
                self.ui.consoleEdit.setTextColor(Qt.red)
                self.ui.consoleEdit.append("GPU_ratio value is over 1 or minus ")
                self.ui.consoleEdit.setTextColor(Qt.black)
                all_data_checked = False
        except ValueError:
            self.ui.consoleEdit.setTextColor(Qt.red)
            self.ui.consoleEdit.append("GPU_ratio value is not numeric or float")
            self.ui.consoleEdit.setTextColor(Qt.black)

        # fine tune check(True or False,doesn't affect data check)
        self.fine_tune = False
        if self.ui.chkB_finetune.isChecked():
            self.fine_tune = True
        self.AI_var["fine_tune"] = self.fine_tune
        self.ui.consoleEdit.append("fine tune is {}".format(self.fine_tune))

        # save ckpt check(True or False,doesn't affect data check)
        self.save_ckpt = True
        if not self.ui.chkB_save_ckpt.isChecked():
            self.save_ckpt = False
        self.AI_var["save_ckpt"] = self.save_ckpt
        self.ui.consoleEdit.append("save ckpt is {}".format(self.save_ckpt))

        return all_data_checked

    def AI_model_finish_process(self):
        self.AI_model.deleteLater()
        self.thread.quit()
        self.ui.consoleEdit.append("AI model execution is finished")
        self.ui.btn_data_process.setEnabled(True)




app = QApplication(sys.argv)
w = AppWindow()
w.show()
sys.exit(app.exec_())
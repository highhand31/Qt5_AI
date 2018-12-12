from PyQt5.QtWidgets import QApplication,QFileDialog,QMainWindow,QGridLayout,QGraphicsScene
from PyQt5.QtCore import QObject,pyqtSignal,QThread,Qt
from AI_test3 import Ui_Form
import sys
from queue import Queue
import json
import socket as sc
import os
import AI_one_class as AI
import common as cm
import matplotlib.image as mpimg
from myFigure import MyFigure
import csv
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import cv2
import time
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class WriteStream(object):
    def __init__(self,queue):
        self.queue = queue

    def write(self, text):
        self.queue.put(text)


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
        try:
            print("Start train data pre-process\n")
            (x_train, y_train_label, no1, no2) = cm.data_load(self.train_dir_name, train_ratio=self.train_ratio,
                                                              resize=(self.resize_width,self.resize_height),
                                                              has_dir=self.train_has_dir)
            print('Training data shape = {}'.format(x_train.shape))

            print("Start test data pre-process\n")
            (x_train_2, y_train_label_2, x_test, y_test_label) = cm.data_load(self.test_dir_name, train_ratio=0,
                                                              resize=(self.resize_width, self.resize_height),
                                                              has_dir=self.test_has_dir)
            print('Test data shape = {}'.format(x_test.shape))
            print('Test label shape = {}'.format(y_test_label.shape))
        except:
            print("在進行訓練資料前處理時發生錯誤")

        self.ae = AI.AE()
        self.ae.train(x_train,x_test,y_test_label,self.GPU_ratio,self.epoch,
                      self.batch_size,self.fine_tune,self.save_ckpt)

        self.finish_flag.emit()

class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # UI init
        #-->Tab:train
        self.ui.label_ver.setText("Version: 1.0.1")
        self.ui.batchEdit.setHidden(True)  # 隱藏，因為使用unpool_with_argmax，batch size限制在1
        self.ui.label_15.setHidden(True)  # 隱藏，因為使用unpool_with_argmax，batch size限制在1
        #self.ui.chkB_finetune.setHidden(True)#不提供fine tune

            # setup train graphics
        self.Fig_train = MyFigure(width=4, height=4)  # 顯示training accuracy
            #在GUI的graphicsView上建立QGraphicsScene，再其上增加以matplotlib圖形為主的FigureCanvas物件
        # self.scene_2 = QGraphicsScene()  # 建立場景QGraphicsScene
        # self.scene_2.addWidget(self.Fig_train)  # 將圖形元素添加到場景中
        # self.ui.graphV_train.setScene(self.scene_2)  # 將場景添加至graphicsView中
        # self.ui.graphV_train.show()  # 顯示
        self.plot_train = self.Fig_train.fig.add_subplot(1, 1, 1)  # 在圖形元素中添加圖

        self.plot_train_toolbar = NavigationToolbar(self.Fig_train, self)  # 添加完整的 toolbar
        self.plot_train_layout = QGridLayout(self.ui.graphV_train)  # 繼承容器groupBox
        self.plot_train_layout.addWidget(self.Fig_train)#將Fig加至graphicsView上
        self.plot_train_layout.addWidget(self.plot_train_toolbar)#將toolbar加至graphicsView上

        # self.plot_train.axis("off")

        #-->Tab:exam
        self.ui.btn_detect.setEnabled(False)#未有資料夾路徑前要disable
        self.ui.btn_prev.setEnabled(False)#未有資料夾路徑前要disable
        self.ui.btn_next.setEnabled(False)#未有資料夾路徑前要disable

            # setup exam graphics
        self.Fig_normal = MyFigure(width=4, height=4)  # 顯示normal pic
        # self.Fig_exam = MyFigure(width=3, height=3)  # 顯示exam pic

            # 在GUI的graphicsView上建立QGraphicsScene，再其上增加以matplotlib圖形為主的FigureCanvas物件
        self.scene_1 = QGraphicsScene()  # 建立場景QGraphicsScene
        self.scene_1.addWidget(self.Fig_normal)  # 將圖形元素添加到場景中
        self.ui.graphV_normal.setScene(self.scene_1)  # 將場景添加至graphicsView中
        self.ui.graphV_normal.show()  # 顯示
        self.plot_normal = self.Fig_normal.fig.add_subplot(1, 1, 1)  # 在圖形元素中添加圖
        self.plot_normal.axis("off")

        # variables init
        self.train_dir_name = self.ui.train_dir_display.placeholderText()
        self.test_dir_name = self.ui.test_dir_display_2.placeholderText()
        self.exam_pb_addr = ""
        self.binary_pb = False
        self.binary_pic_selection = False
        self.AI_var = {}
        self.epoch = self.ui.epochEdit.text()
        self.resize_height = int(self.ui.heightEdit.text())
        self.resize_width = int(self.ui.widthEdit.text())
        self.GPU_ratio = float(self.ui.gpuratioEdit.text())
        # print(self.resize_height)
        # print(type(self.resize_height))
        # self.batch_size = self.ui.batchEdit.text()
        self.batch_size = 1

        # sys.stdout change to queue
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

        # thread3:UDP listener
        self.thread_UDP_listener = QThread()
        self.UDP_listen = UDP_listener()
        self.UDP_listen.UDP_signal.connect(self.UDP_get_data)
        self.UDP_listen.moveToThread(self.thread_UDP_listener)
        self.thread_UDP_listener.started.connect(self.UDP_listen.run)
        self.thread_UDP_listener.start()



        #print("測試OK")

        # setup events
        self.ui.btn_train_dir.clicked.connect(self.btn_train_dir_clicked)
        self.ui.btn_test_dir.clicked.connect(self.btn_test_dir_clicked)
        self.ui.btn_data_process.clicked.connect(self.btn_AI_training_clicked)
        #-->Tab:exam
        self.ui.btn_exam_dir.clicked.connect(self.btn_exam_dir_clicked)
        self.ui.btn_exam_pb_dir.clicked.connect(self.btn_exam_pb_dir_clicked)
        self.ui.btn_prev.clicked.connect(self.btn_prev_clicked)
        self.ui.btn_next.clicked.connect(self.btn_next_clicked)
        self.ui.btn_detect.clicked.connect(self.btn_detect_clicked)

        # UI plot code(temp)
        # add by Johnny
        # self.matplotlibwidget_dynamic = MatplotlibWidget(self.tab)
        # self.matplotlibwidget_dynamic.setEnabled(True)
        # self.matplotlibwidget_dynamic.setHidden(True)
        # self.matplotlibwidget_dynamic.setGeometry(QtCore.QRect(0, 500, 620, 450))
        # self.matplotlibwidget_dynamic.setObjectName("matplotlibwidget_dynamic")

    def btn_train_dir_clicked(self):
        self.train_dir_name = QFileDialog.getExistingDirectory(self, "select Train data directory")
        self.ui.train_dir_display.setText(self.train_dir_name)
        #self.train_dir_name = repr(self.train_dir_name)[1:-1]

        # if self.train_dir_name != "":
        #     self.ui.train_dir_display.setText(self.train_dir_name)
        #
        # else:
        #     print("沒有選擇到資料夾，請重新選擇")

    def btn_test_dir_clicked(self):
        self.test_dir_name = QFileDialog.getExistingDirectory(self, "select Test data directory")
        #self.test_dir_name = repr(self.test_dir_name)[1:-1]
        self.ui.test_dir_display_2.setText(self.test_dir_name)

    def btn_exam_dir_clicked(self):
        # reset flag
        self.binary_pic_selection = False
        #disable btn
        self.ui.btn_prev.setEnabled(False)
        self.ui.btn_next.setEnabled(False)
        self.ui.btn_detect.setEnabled(False)
        # clear display
        self.ui.textEdit_exam.clear()
        #選擇資料夾
        self.exam_dir_name = QFileDialog.getExistingDirectory(self, "select Exam data directory")

        if not self.exam_dir_name == "":
            # self.exam_dir_name = "r" + "'" + self.exam_dir_name + "'"
            #self.exam_dir_name = unicode(self.exam_dir_name, 'utf8')
            #self.exam_dir_name = repr(self.exam_dir_name)[1:-1]
            self.ui.exam_dir_display.setText(self.exam_dir_name)

            #紀錄所有圖片的路徑
                #先記錄資料夾內所有"檔案"的路徑
            try:
                # 確認資料夾裡是否有圖片
                self.exam_pic_num, self.exam_pic_addrs = self.pic_addr_check(self.exam_dir_name)
                if self.exam_pic_num < 1:
                    self.ui.consoleEdit.setTextColor(Qt.red)
                    self.ui.textEdit_exam.append("訓練資料夾裡沒有圖片")
                    self.ui.consoleEdit.setTextColor(Qt.black)

                    self.plot_normal.cla()#清除之前的圖片
                    self.plot_normal.axis("off")
                    self.Fig_normal.draw()  # 一定要這行才能顯示出圖片

                else:
                    self.index = 0  # 預設圖片index
                    # self.exam_pic_num = len(self.exam_pic_addrs)
                    #print("Picture number of dir({}) is {} ".format(self.exam_dir_name, self.exam_pic_num))
                    # display the pic
                    # self.img = cv2.imread(self.exam_pic_addrs[0])
                    # self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
                    # 使用cv2.imread容易因為路徑有\的相關問題，改用mpimg
                    self.img = mpimg.imread(self.exam_pic_addrs[0])
                    self.plot_normal.imshow(self.img)
                    self.plot_normal.axis("off")
                    self.Fig_normal.draw()  # 一定要這行才能顯示出圖片

                    #圖片選擇成功
                    self.binary_pic_selection = True
                    #確定有圖片可顯示再enable按鈕
                    self.ui.btn_prev.setEnabled(True)  # 未有資料夾路徑前要disable
                    self.ui.btn_next.setEnabled(True)  # 未有資料夾路徑前要disable

                    #確認PB檔是否已經選擇完成
                    if self.binary_pb is True:
                        self.ui.btn_detect.setEnabled(True)

            except:
                self.ui.textEdit_exam.append("錯誤發生")

        else:
            self.ui.textEdit_exam.append("沒有選擇到資料夾，請重新選擇")

    def btn_exam_pb_dir_clicked(self):
        #reset flag
        self.binary_pb = False
        #disable btn
        self.ui.btn_detect.setEnabled(False)
        # clear display
        self.ui.textEdit_exam.clear()
        #選擇pb檔
        self.exam_pb_addr,filetype = QFileDialog.getOpenFileName(self,"select Exam Model file","./","PB Files (*.pb)")

        if not self.exam_pb_addr == "":
            # self.exam_dir_name = "r" + "'" + self.exam_dir_name + "'"
            #self.exam_dir_name = unicode(self.exam_dir_name, 'utf8')
            #self.exam_dir_name = repr(self.exam_dir_name)[1:-1]
            self.ui.exam_pb_display.setText(self.exam_pb_addr)

            try:
                # 讀取PB黨
                # PB檔位址
                # model_filename = "model_saver/pb_test_model.pb"
                model_filename = self.exam_pb_addr
                # 設定GPU參數
                config = tf.ConfigProto(log_device_placement=True,
                                        allow_soft_placement=True,  # 允許當找不到設備時自動轉換成有支援的設備
                                        )
                config.gpu_options.per_process_gpu_memory_fraction = 0.1
                with tf.Session(config=config) as sess:
                    with gfile.FastGFile(model_filename, 'rb') as f:
                        graph_def = tf.GraphDef()
                        graph_def.ParseFromString(f.read())
                        sess.graph.as_default()

                        tf.import_graph_def(graph_def, name='')  # 導入計算圖
                        # self.ui.textEdit_exam.append("ok1")
                    self.input_x = sess.graph.get_tensor_by_name("input_x:0")
                    self.loss = sess.graph.get_tensor_by_name("loss:0")
                    self.result = sess.graph.get_tensor_by_name("output/Relu:0")

                self.ui.textEdit_exam.append("模型載入成功")
                #PB檔載入成功
                self.binary_pb = True
                #enable btn
                if self.binary_pic_selection is True:
                    self.ui.btn_prev.setEnabled(True)
                    self.ui.btn_next.setEnabled(True)
                    self.ui.btn_detect.setEnabled(True)

            except:
                self.ui.textEdit_exam.append("錯誤發生，可能是檔案與訓練時的模型不相同")

        else:
            self.ui.textEdit_exam.append("沒有選擇到資料夾，請重新選擇")


    def btn_next_clicked(self):
        #disable btn
        self.ui.btn_next.setEnabled(False)
        #clear label_exam_warn
        self.ui.label_exam_warn.clear()
        #index
        self.index += 1
        # if self.number > len(self.files):
        self.index = min(self.index, len(self.exam_pic_addrs) - 1)
        if self.index == len(self.exam_pic_addrs) - 1:
            self.ui.label_exam_warn.setText("已是最後1張圖片")

        #display the pic
        # self.img = cv2.imread(self.exam_pic_addrs[self.index])
        # self.img = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
        self.img = mpimg.imread(self.exam_pic_addrs[self.index])
        self.plot_normal.imshow(self.img)
        self.plot_normal.axis("off")
        self.Fig_normal.draw()  # 一定要這行才能顯示出圖片

        # self.image.load(self.files[self.number])
        # self.LoadImage()
        self.ui.btn_next.setEnabled(True)

    def btn_prev_clicked(self):
        self.ui.btn_prev.setEnabled(False)
        # clear label_exam_warn
        self.ui.label_exam_warn.clear()
        self.index -= 1
        self.index = max(0,self.index)
        if self.index == 0:
            self.ui.label_exam_warn.setText("已是第1張圖片")

        # display the pic
        # self.img = cv2.imread(self.exam_pic_addrs[self.index])
        # self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.img = mpimg.imread(self.exam_pic_addrs[self.index])
        self.plot_normal.imshow(self.img)
        self.plot_normal.axis("off")
        self.Fig_normal.draw()  # 一定要這行才能顯示出圖片

        self.ui.btn_prev.setEnabled(True)
    def btn_detect_clicked(self):
        # 清除之前的文字內容
        self.ui.textEdit_exam.clear()
        #PB檔位址
        model_filename = "model_saver/pb_test_model.pb"
        #暫停按鈕功能
        self.ui.btn_next.setEnabled(False)
        self.ui.btn_prev.setEnabled(False)
        self.ui.btn_detect.setEnabled(False)

        #檢驗GPU Ratio
        if self.GPU_ratio >= 0.9:
            self.ui.textEdit_exam.setText("訓練GPU ratio已經使用{}，GPU資源不夠進行圖片驗證".format(self.GPU_ratio))
        else:
            GPU_ratio_exam = 0.1
            #self.ui.textEdit_exam.setText("圖片驗證的GPU ratio = {}".format(GPU_ratio_exam))

            #設定GPU參數
            config = tf.ConfigProto(log_device_placement=True,
                                    allow_soft_placement=True,  # 允許當找不到設備時自動轉換成有支援的設備
                                    )
            config.gpu_options.per_process_gpu_memory_fraction = GPU_ratio_exam
            # with tf.Session(config=config) as sess:
            #     with gfile.FastGFile(model_filename, 'rb') as f:
            #         graph_def = tf.GraphDef()
            #         graph_def.ParseFromString(f.read())
            #         sess.graph.as_default()
            #
            #         tf.import_graph_def(graph_def, name='')  # 導入計算圖
            #         # self.ui.textEdit_exam.append("ok1")
            #     input_x = sess.graph.get_tensor_by_name("input_x:0")
            #     loss = sess.graph.get_tensor_by_name("loss:0")
            #     result = sess.graph.get_tensor_by_name("output/Relu:0")
            with tf.Session(config=config) as sess:


                sess.run(tf.global_variables_initializer())

                # self.ui.textEdit_exam.append("ok2")

                pb_height = self.result.shape[1]
                pb_width = self.result.shape[2]
                # self.ui.textEdit_exam.append("ok3")

                #picture process
                self.img = cv2.resize(self.img, (pb_width, pb_height))
                input_test = []
                input_test.append(self.img)
                input_test = np.array(input_test)
                input_test.astype("float32")
                input_test = input_test / 255
                #tf.Session execution
                prediction = sess.run(self.result, feed_dict={self.input_x: input_test[0:1]})#一次僅能投一張圖片
                predict_loss = sess.run(self.loss,feed_dict={self.input_x: input_test[0:1]})

                #picture display
                # diff = np.abs(prediction - input_test)
                # self.plot_defect.imshow(diff[0])
                # self.plot_defect.axis("off")
                # self.Fig_exam.draw()  # 一定要這行才能顯示出圖片

                #read the train loss from csv file
                file_name = "train_notes.csv"
                with open(file_name) as csvFile:
                    # fields = ["average loss"]
                    dictReader = csv.DictReader(csvFile)
                    for item in dictReader:
                        train_loss = item["average loss"]
                        train_stdv = item["stdv"]

                try:
                    train_loss = float(train_loss)
                    train_stdv = float(train_stdv)
                    # self.ui.textEdit_exam.append("average loss = {}".format(train_loss))
                    # self.ui.textEdit_exam.append("standard deviation= {}".format(train_stdv))
                    # self.ui.textEdit_exam.append("exam pic loss = {}".format(predict_loss))
                    # self.ui.textEdit_exam.append("average loss type= {}".format(type(train_loss)))
                    predict_loss = float(predict_loss)
                    # self.ui.textEdit_exam.append("exam pic loss type= {}".format(type(predict_loss)))

                    self.ui.textEdit_exam.append("The time is {}\n".format(time.asctime()))
                    if predict_loss <= train_loss:#+1*train_stdv:
                        self.ui.textEdit_exam.append("Good")
                    else:
                        self.ui.textEdit_exam.append("NG")

                except ValueError:
                    self.ui.textEdit_exam.append("train loss is not float value")


                # if predict_loss > train_loss:
                #     self.ui.textEdit_exam.append("The picture is good")
                #
                # else:
                #     self.ui.textEdit_exam.append("The picture is NG")
        # 開啟按鈕功能
        self.ui.btn_next.setEnabled(True)
        self.ui.btn_prev.setEnabled(True)
        self.ui.btn_detect.setEnabled(True)

    def btn_AI_training_clicked(self):
        # self.ui.consoleEdit.setGeometry(QtCore.QRect(0, 440, 385, 301))#left, top, width and height

        #set plot visible
        #self.ui.matplotlibwidget_dynamic.setHidden(False)
        #self.ui.matplotlibwidget_dynamic.mpl.start_static_plot()
        # self.fig = self.ui.matplotlibwidget_dynamic.mpl.fig
        # self.axes = self.ui.matplotlibwidget_dynamic.mpl.axes
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
            self.ui.consoleEdit.setTextColor(Qt.black)
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
            self.ui.consoleEdit.setTextColor(Qt.red)
            self.ui.consoleEdit.append("UI inputs are not completed, plz see msg above\n")
            self.ui.consoleEdit.setTextColor(Qt.black)
            self.ui.btn_data_process.setEnabled(True)


    def AI_param_check(self):
        all_data_checked = True
        self.ui.consoleEdit.append("Start to exam UI inputs")

        # train dir check
            #確認是否有資料夾名稱
        if self.train_dir_name == "":
            self.ui.consoleEdit.setTextColor(Qt.red)
            self.ui.consoleEdit.append("No train dir")
            self.ui.consoleEdit.setTextColor(Qt.black)
            all_data_checked = False
        else:

            #確認資料夾裡是否有圖片
            pic_num,pic_addr = self.pic_addr_check(self.train_dir_name)
            if pic_num < 1:
                self.ui.consoleEdit.setTextColor(Qt.red)
                self.ui.consoleEdit.append("訓練資料夾裡沒有圖片")
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
            #確認test資料夾檔名是否為空
        if self.test_dir_name == "":
            self.ui.consoleEdit.setTextColor(Qt.red)
            self.ui.consoleEdit.append("No test dir")
            self.ui.consoleEdit.setTextColor(Qt.black)
            all_data_checked = False
        else:
            #確認資料夾裡是否有2個資料夾
            dir_test = [file.path for file in os.scandir(self.test_dir_name) if file.is_dir()]
            if dir_test == []:
                self.ui.consoleEdit.setTextColor(Qt.red)
                self.ui.consoleEdit.append("驗證資料夾裡沒有分類好的資料夾")
                self.ui.consoleEdit.setTextColor(Qt.black)
                all_data_checked = False
            elif len(dir_test)<2:
                self.ui.consoleEdit.setTextColor(Qt.red)
                self.ui.consoleEdit.append("驗證資料夾裡請至少有要2個資料夾(Good and NG for example)")
                self.ui.consoleEdit.setTextColor(Qt.black)
                all_data_checked = False

            else:
                check_flag = True
                for dir in dir_test:
                    pic_num, pic_addrs = self.pic_addr_check(dir)
                    if pic_num < 1:
                        self.ui.consoleEdit.setTextColor(Qt.red)
                        self.ui.consoleEdit.append("dir({})沒有圖片資料".format(dir))
                        self.ui.consoleEdit.setTextColor(Qt.black)
                        all_data_checked = False
                        check_flag = False

                if check_flag is True:
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
                if self.GPU_ratio == 1.0:
                    self.GPU_ratio = 0.9#限制GPU使用量不會超過0.9
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

    def append_text(self,text):
        self.ui.consoleEdit.append(text)

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
                self.Fig_train.fig.suptitle('Accuracy of train data')
                #self.fig.suptitle('Loss and Accuracy')
                #l = [np.random.randint(0, 10) for i in range(4)]
                #self.axes.plot([0, 1, 2, 3], l, 'r')
                #去除上一次的數值
                self.plot_train.cla()
                # self.axes.cla()

                #繪製線條
                epoch_num = len(text["acc"])
                axis_x = [num for num in range(1,epoch_num+1)]#畫圖的X軸(epoch)要從1開始
                #self.axes.plot(text["train loss"],label="train loss")
                self.plot_train.plot(axis_x,text["acc"],label="accuracy")
                # self.axes.plot(text["acc"],label="accuracy")
                #設置圖例位置

                self.plot_train.legend(loc="best", shadow=True)
                self.plot_train.set_ylabel('accuracy')
                self.plot_train.set_xlabel('epoch')
                self.plot_train.grid(True)
                self.Fig_train.draw()
                # self.ui.matplotlibwidget_dynamic.mpl.draw()

        except ValueError:
            print("UDP receives messages with error format")

    def pic_addr_check(self,pic_path):
        # 建立圖片格式的集合
        self.pic_format = {".jpg", ".jpeg", ".jpe", ".bmp", ".dib", ".jp2", ".png", ".webp", ".pbm",
                           ".pgm", ".ppm", ".sr", ".ras", ".tiff", ".tif"}
        temp = [file.path for file in os.scandir(pic_path) if file.is_file()]
        pic_addrs = []
        pic_num = 0
        for item in temp:
            # 分離路徑的副檔名，[-1]表示檔案的格式
            item_format = os.path.splitext(item)[-1]
            # 檢查檔案格式是否屬於預設的圖片格式
            if item_format in self.pic_format:
                pic_addrs.append(item)
                pic_num += 1

        return (pic_num,pic_addrs)

    def AI_model_finish_process(self):
        self.AI_model.deleteLater()
        self.thread.quit()
        self.ui.consoleEdit.setTextColor(Qt.blue)
        self.ui.consoleEdit.append("AI model execution is finished")
        self.ui.consoleEdit.setTextColor(Qt.black)
        self.ui.btn_data_process.setEnabled(True)

app = QApplication(sys.argv)
w = AppWindow()
w.setFixedSize(641,1000)#固定UI的畫面大小
w.show()
sys.exit(app.exec_())
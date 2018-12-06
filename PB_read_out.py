import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.platform import gfile

model_filename = "model_saver/pb_test_model.pb"

pic_path = r".\xxx\Crack\pill_magnesium_crack_312.png"
img = cv2.imread(pic_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (64, 64))
input_test = []
input_test.append(img)
input_test = np.array(input_test)
input_test.astype(np.float32)
input_test = input_test / 255
# 設定GPU參數
config = tf.ConfigProto(log_device_placement=True,
                        allow_soft_placement=True,  # 允許當找不到設備時自動轉換成有支援的設備
                        )
config.gpu_options.per_process_gpu_memory_fraction = 0.5
with tf.Session() as sess:
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()

        tf.import_graph_def(graph_def, name='')  # 導入計算圖

    sess.run(tf.global_variables_initializer())
    input_x = sess.graph.get_tensor_by_name("input_x:0")
    print(input_x.shape)
    result = sess.graph.get_tensor_by_name("output/Relu:0")
    print(result.shape)

    a = sess.run(result, feed_dict={input_x: input_test[0:1]})
    print(a)
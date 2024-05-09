"""---------------------------------------------------------
四、人脸识别
1、打开摄像头，获取图片并灰度化
2、人脸检测
3、加载卷积神经网络模型
4、人脸识别
------------------------------------------------------------"""
import tensorflow as tf
import cv2
import dlib
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split
import net
import units


"""定义人脸识别函数"""
def face_recognise(image):
    res = sess.run(predict, feed_dict={input_image: [image/255.0], dropout_rate: 1.0, dropout_rate_2: 1.0})
    if res[0] == 1:
        return "my_face"
    else:
        return "other_face"

if __name__ == '__main__':
    """定义参数"""
    num_class =2
    size = 64  # 图片大小64*64*3
    input_image = tf.placeholder(tf.float32, [None, size, size, 3])  # 输入X：64*64*3  ， 定义
    input_label = tf.placeholder(tf.float32, [None, 2])  # 输出Y_：1*2  ， 定义
    dropout_rate = tf.placeholder(tf.float32)  # 定义
    dropout_rate_2 = tf.placeholder(tf.float32)  # 定义

    """将读取的人脸图片输出神经网络，输出out(1,2)"""
    outdata = net.layer_net(input_image, num_class, dropout_rate, dropout_rate_2)  # outdata: [0.3,0.7] , input_label: [1, 0]
    """预测， 1表示按行返回out中最大值的索引，而不是out与1比谁大返回谁，predict为索引值，0或1，因为out的shape是（1,2）,一行两列，两个数字"""
    predict = tf.argmax(outdata, 1)   # 0或者1

    """1、读取模型文件"""
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint('G:/github/CNN_faces_recognition/model_two/'))
    """2、检测人脸，特征提取器: dlib自带的frontal_face_detector"""
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(0)                  # 打开摄像头
    while True:
        _, img = cap.read()                    # 读取
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      # 灰度化
        dets = detector(gray_image, 1)
        if not len(dets):
            key = cv2.waitKey(30)
            if key == 27:
                sys.exit(0)
        """--------------------------------------------------------------------
        使用enumerate 函数遍历序列中的元素以及它们的下标,i为人脸序号,d为i对应的元素;
        left：人脸左边距离图片左边界的距离 ；right：人脸右边距离图片左边界的距离 
        top：人脸上边距离图片上边界的距离 ；bottom：人脸下边距离图片上边界的距离
         ----------------------------------------------------------------------"""
        for i, d in enumerate(dets):
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0
            """人脸大小64*64"""
            face = img[x1:y1, x2:y2]
            face = cv2.resize(face, (size, size))
            """3、人脸识别，输出识别结果"""
            print(' who? %s' % face_recognise(face))
            """4、显示识别结果"""
            cv2.rectangle(img, (x2, x1), (y2, y1), (255, 0, 0), 3)
            if face_recognise(face) == "my_face":
                cv2.putText(img, 'my_face', (x2, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(img, 'other_face', (x2, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('image', img)
        key = cv2.waitKey(30)
        if key == 27:
            sys.exit(0)

    sess.close()

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

"""定义参数"""
faces_my_path = './faces_my'
faces_other_path = './faces_other'
batch_size = 128          # 每次取100张图片
learning_rate = 0.01        # 学习率
size = 64                 # 图片大小64*64*3
imgs = []                 # 存放人脸图片
labs = []                 # 存放人脸图片对应的标签
x = tf.placeholder(tf.float32, [None, size, size, 3])  # 输入X：64*64*3
y_ = tf.placeholder(tf.float32, [None, 2])  # 输出Y_：1*2
keep_prob_fifty = tf.placeholder(tf.float32)  # 50%，即0.5
keep_prob_seventy_five = tf.placeholder(tf.float32)  # 75%，即0.75

"""定义读取人脸数据函数"""
def readData(path , h = size , w = size):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename
            img = cv2.imread(filename)
            top,bottom,left,right = getPaddingSize(img)
            """放大图片扩充图片边缘部分"""
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            img = cv2.resize(img, (h, w))
            imgs.append(img)                 # 一张张人脸图片加入imgs列表中
            labs.append(path)                # 一张张人脸图片对应的path，即文件夹名faces_my和faces_other，即标签

"""定义padding大小函数，有一套公式"""
def getPaddingSize(img):
    height, width, _ = img.shape
    top, bottom, left, right = (0, 0, 0, 0)
    longest = max(height, width)

    if width < longest:
        tmp = longest - width
        left = tmp // 2
        right = tmp - left
    elif height < longest:
        tmp = longest - height
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right

"""定义神经网络层，共五层，卷积层特征提取，池化层降维,全连接层进行分类，共两类：我的人脸（true),不是我的人脸（false）"""
def cnnLayer():
    """第一、二层，输入图片64*64*3，输出图片32*32*32"""
    W1 = tf.Variable(tf.random_normal([3, 3, 3, 32]))                 # 卷积核大小(3,3)， 输入通道(3)， 输出通道(32)
    b1 = tf.Variable(tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='SAME')+b1)    # 64*64*32，卷积提取特征，增加通道数
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 32*32*32，池化降维，减小复杂度
    drop1 = tf.nn.dropout(pool1, keep_prob_fifty)

    """第三、四层，输入图片32*32*32，输出图片16*16*64"""
    W2 = tf.Variable(tf.random_normal([3, 3, 32, 64]))  # 卷积核大小(3,3)， 输入通道(32)， 输出通道(64)
    b2 = tf.Variable(tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.conv2d(drop1, W2, strides=[1, 1, 1, 1], padding='SAME') + b2)        # 32*32*64
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')       # 16*16*64
    drop2 = tf.nn.dropout(pool2, keep_prob_fifty)

    """第五、六层，输入图片16*16*64，输出图片8*8*64"""
    W3 = tf.Variable(tf.random_normal([3, 3, 64, 64]))  # 卷积核大小(3,3)， 输入通道(64)， 输出通道(64)
    b3 = tf.Variable(tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.conv2d(drop2, W3, strides=[1, 1, 1, 1], padding='SAME') + b3)        # 16*16*64
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')       # 8*8*64=4096
    drop3 = tf.nn.dropout(pool3, keep_prob_fifty)

    """第七层，全连接层，输入图片8*8*64，reshape到1*4096，输出1*512"""
    Wf = tf.Variable(tf.random_normal([8*8*64,512]))     # 输入通道(4096)， 输出通道(512)
    bf = tf.Variable(tf.random_normal([512]))
    drop3_flat = tf.reshape(drop3, [-1, 8*8*64])         # -1表示行随着列的需求改变，1*4096
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)   # [1,4096]*[4096,512]=[1,512]
    dropf = tf.nn.dropout(dense, keep_prob_seventy_five)

    """第八层，输出层，输入1*512，输出1*2，再add，输出一个数"""
    Wout = tf.Variable(tf.random_normal([512,2]))        # 输入通道(512)， 输出通道(2)
    bout = tf.Variable(tf.random_normal([2]))
    out = tf.add(tf.matmul(dropf, Wout), bout)     # (1,512)*(512,2)=(1,2) ,跟y_ [0,1]、[1,0]比较给出损失
    return out

"""定义人脸识别函数"""
def face_recognise(image):
    res = sess.run(predict, feed_dict={x: [image/255.0], keep_prob_fifty: 1.0, keep_prob_seventy_five: 1.0})
    if res[0] == 1:
        return "Yes,my face"
    else:
        return "No,other face"

if __name__ == '__main__':

    """1、读取人脸数据"""
    readData(faces_my_path)
    readData(faces_other_path)
    imgs = np.array(imgs)  # 将图片数据与标签转换成数组
    labs = np.array([[0, 1] if lab == faces_my_path else [1, 0] for lab in labs])  # 标签：[0,1]表示是我的人脸，[1,0]表示其他的人脸
    """2、随机划分测试集与训练集"""
    train_x_1, test_x_1, train_y, test_y = train_test_split(imgs, labs, test_size=0.05,
                                                            random_state=random.randint(0, 100))
    train_x_2 = train_x_1.reshape(train_x_1.shape[0], size, size, 3)  # 参数：图片数据的总数，图片的高、宽、通道
    test_x_2 = test_x_1.reshape(test_x_1.shape[0], size, size, 3)
    train_x = train_x_2.astype('float32') / 255.0                      # 归一化
    test_x = test_x_2.astype('float32') / 255.0
    print('Train Size:%s, Test Size:%s' % (len(train_x), len(test_x)))
    num_batch = len(train_x) // batch_size                    # 22784//128=178
    """3、将读取的人脸图片输出神经网络，输出out(1,2)"""
    out = cnnLayer()
    """4、预测， 1表示按行返回out中最大值的索引，而不是out与1比谁大返回谁，predict为索引值，0或1，因为out的shape是（1,2）,一行两列，两个数字"""
    predict = tf.argmax(out, 1)   # 0或者1

    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    """5、检测人脸，特征提取器: dlib自带的frontal_face_detector"""
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
            """6、人脸识别，输出识别结果"""
            print('It recognizes my face? %s' % face_recognise(face))

            cv2.rectangle(img, (x2, x1), (y2, y1), (255, 0, 0), 3)
            if face_recognise(face) == "Yes,my face":
                cv2.putText(img, 'Yes,my face', (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(img, 'No,other face', (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
            """通过确定对角线画矩形"""
            #cv2.rectangle(img, (x2, x1), (y2, y1), (255, 0, 0), 3)
        cv2.imshow('image', img)
        key = cv2.waitKey(30)
        if key == 27:
            sys.exit(0)

    sess.close()

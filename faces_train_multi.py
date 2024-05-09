"""--------------------------------------------------------------
三、CNN模型训练
训练模型：共八层神经网络，卷积层特征提取，池化层降维,全连接层进行分类。
训练数据：22784，测试数据：727，训练集：测试集=20:1
共两类：我的人脸（yes),不是我的人脸（no）。
共八层： 第一、二层（卷积层1、池化层1），输入图片64*64*3，输出图片32*32*32
        第三、四层（卷积层2、池化层2），输入图片32*32*32，输出图片16*16*64
        第五、六层（卷积层3、池化层3），输入图片16*16*64，输出图片8*8*64
        第七层（全连接层），输入图片8*8*64，reshape到1*4096，输出1*512
        第八层（输出层），输入1*512，输出1*2
学习率：0.01
损失函数：交叉熵
优化器：Adam
------------------------------------------------------------------"""
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import shutil
import random
import os
import time
import matplotlib.pyplot as plt
import sys
import net
import units


"""定义读取人脸数据函数，根据不同的人名，分配不同的onehot值"""
def get_images_labels(in_path , height , width):
    for file in os.listdir(in_path):
        if file.endswith('.jpg'):
            file = in_path + '/' + file
            img = cv2.imread(file)
            t, b, l, r = units.img_padding(img)
            """放大图片扩充图片边缘部分"""
            img_big = cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            img_big = cv2.resize(img_big, (height, width))
            imgs.append(img_big)                 # 一张张人脸图片加入imgs列表中
            labs.append(in_path)             # 一张张人脸图片对应的path，即文件夹名faces_my和faces_other，即标签


"""定义训练函数"""
def do_train(outdata, cross_entropy, optimizer):
    """求得准确率，比较标签是否相等，再求的所有数的平均值"""
    """tf.argmax()返回最大索引值，[1, 0] -> 1大-> 索引值0"""
    """tf.equal()判断是否相等，返回True or False"""
    """tf.cast()函数用于执行tensorflow中张量数据类型转换。"""
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(outdata, 1), tf.argmax(input_label, 1)), tf.float32))
    print("accuracy : ", accuracy)

    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()

    config = tf.ConfigProto(allow_soft_placement=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config.gpu_options.allow_growth = True
    steps = []
    losss = []
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter('./tmp', graph=tf.get_default_graph())
        for n in range(100):
            # """每次取100(batch_size)张图片,num_batch = len(train_x) // batch_size = 227"""
            # loss是拿训练集算的
            for i in range(num_batch):
                batch_x = train_x_normalization[i*batch_size: (i+1)*batch_size]          # 图片
                batch_y = train_y[i*batch_size: (i+1)*batch_size]          # 标签：[0,0,0,1] [1,0,0,0]...
                _, loss, summary = sess.run([optimizer, cross_entropy, merged_summary_op],
                                            feed_dict={input_image: batch_x, input_label: batch_y,
                                                       dropout_rate: 1, dropout_rate_2: 1})
                summary_writer.add_summary(summary, n*num_batch+i)
                steps.append(str(n * num_batch + i))
                losss.append(str(loss))
                print("step:%d,  loss:%g" % (n * num_batch + i, loss))
                units.write_txt('./model_multi/loss.txt', losss)

                if (n*num_batch+i) % 100 == 0:
                    acc = accuracy.eval({input_image: test_x_normalization, input_label: test_y, dropout_rate: 1.0, dropout_rate_2: 1.0})
                    print("step:%d,  acc:%g" % (n*num_batch+i, acc))
                    saver.save(sess, './model_multi/train_faces.model', global_step=n*num_batch+i)



if __name__ == '__main__':
    """定义参数"""

    faces_my_path = './faces_my'         # [0,0,0,1]
    faces_sxx_path = './faces_sxx'       # [0,0,1,0]
    faces_wtt_path = './faces_wtt'       # [0,1,0,0]
    faces_other_path = './faces_other'   # [1,0,0,0]
    num_class = 4
    batch_size = 128  # 每次取128张图片        100   128
    learning_rate = 0.01  # 学习率   0.01    0.1

    size = 64  # 图片大小64*64*3
    imgs = []  # 存放人脸图片
    labs = []  # 存放人脸图片对应的标签

    """1、读取人脸数据、分配标签"""
    get_images_labels(faces_my_path, size, size)
    get_images_labels(faces_sxx_path, size, size)
    get_images_labels(faces_wtt_path, size, size)
    get_images_labels(faces_other_path, size, size)

    imgs = np.array(imgs)                   # 将图片数据与标签转换成数组
    print("len imgs: ", len(imgs))
    # 标签：[0, 0, 0, 1]表示zsc的人脸，[0, 0, 1, 0]表示sxx的人脸, [0, 1, 0, 0]表示wtt的人脸, [1, 0, 0, 0]表示other的人脸
    for id, lab in enumerate(labs):
        if lab == faces_my_path:
            labs[id] = [0, 0, 0, 1]
        elif lab == faces_sxx_path:
            labs[id] = [0, 0, 1, 0]
        elif lab == faces_wtt_path:
            labs[id]= [0, 1, 0, 0]
        else:
            labs[id] = [1, 0, 0, 0]
    labs = np.array(labs)

    print("labs: ", labs)
    print("len labs: ", len(labs))


    """2、随机划分测试集与训练集,按照1：20的比列"""
    train_x, test_x, train_y, test_y = train_test_split(imgs, labs, test_size=1 / 20,
                                                        random_state=random.randint(0, 100))
    print("[train_x, test_x, train_y, test_y]: ", [len(train_x), len(test_x), len(train_y), len(test_y)])
    train_x_reshape = train_x.reshape(train_x.shape[0], size, size, 3)  # 参数：图片数据的总数，图片的高、宽、通道
    test_x_reshape = test_x.reshape(test_x.shape[0], size, size, 3)
    # print('train_x_reshape: ', train_x_reshape)
    """3、归一化"""
    train_x_normalization = train_x_reshape.astype('float32') / 255.0
    test_x_normalization = test_x_reshape.astype('float32') / 255.0
    # print('train_x_normalization: ', train_x_normalization)
    print('len(train_x_normalization): ', len(train_x_normalization))
    print('len(test_x_normalization): ', len(test_x_normalization))

    # 定义训练参数
    num_batch = len(train_x_normalization) // batch_size  # 22786// 100
    print('num_batch: ', num_batch)
    input_image = tf.placeholder(tf.float32, [None, size, size, 3])  # 输入X：64*64*3  ， 定义
    input_label = tf.placeholder(tf.float32, [None, 4])  # 输出Y_：1*4  ， 定义
    dropout_rate = tf.placeholder(tf.float32)  # 定义
    dropout_rate_2 = tf.placeholder(tf.float32)  # 定义

    """神经网络输出,还没给图片"""
    outdata = net.layer_net(input_image, num_class, dropout_rate, dropout_rate_2)  # outdata: [0.3,0.7] , input_label: [1, 0]
    """定义损失函数为交叉熵"""
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outdata, labels=input_label))
    """采用Adam优化器"""
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    # optimizer = tf.train.MomentumOptimizer(learning_rate).minimize(cross_entropy)

    """4、进行训练"""
    do_train(outdata, cross_entropy, optimizer)

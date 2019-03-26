"""--------------------------------------------------------------
三、CNN模型训练
训练模型：共八层神经网络，卷积层特征提取，池化层降维,全连接层进行分类。
训练数据：22784，测试数据：1200，训练集：测试集=20:1
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
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import random
import cv2
import sys
import os

"""定义参数"""
faces_my_path = './faces_my'
faces_other_path = './faces_other'
batch_size = 100          # 每次取100张图片
learning_rate = 0.01        # 学习率
size = 64                 # 图片大小64*64*3
imgs = []                 # 存放人脸图片
labs = []                 # 存放人脸图片对应的标签

"""定义读取人脸数据函数，根据不同的人名，分配不同的onehot值"""
def readData(path , h = size , w = size):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename
            img = cv2.imread(filename)
            top, bottom, left, right = getPaddingSize(img)
            """放大图片扩充图片边缘部分"""
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            img = cv2.resize(img, (h, w))
            imgs.append(img)                 # 一张张人脸图片加入imgs列表中
            labs.append(path)                # 一张张人脸图片对应的path，即文件夹名faces_my和faces_other，即标签

"""定义尺寸变换函数，有一套公式"""
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
    drop1 = tf.nn.dropout(pool1, keep_prob_fifty)      # 按一定概率随机丢弃一些神经元，以获得更高的训练速度以及防止过拟合

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


    """第七层，全连接层，将图片的卷积输出压扁成一个一维向量，输入图片8*8*64，reshape到1*4096，输出1*512"""
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

"""定义训练函数"""
def train():
    out = cnnLayer()
    """损失函数为交叉熵"""
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_))
    """采用Adam优化器"""
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    """求得准确率，比较标签是否相等，再求的所有数的平均值"""
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1)), tf.float32))

    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter('./tmp', graph=tf.get_default_graph())
        for n in range(10):

             # """每次取100(batch_size)张图片,num_batch = len(train_x) // batch_size"""
            for i in range(num_batch):
                batch_x = train_x[i*batch_size: (i+1)*batch_size]          # 图片
                batch_y = train_y[i*batch_size: (i+1)*batch_size]          # 标签：[0,1] [1,0]
                _, loss, summary = sess.run([optimizer, cross_entropy, merged_summary_op],
                                            feed_dict={x: batch_x, y_: batch_y,
                                                       keep_prob_fifty: 0.5, keep_prob_seventy_five: 0.75})
                summary_writer.add_summary(summary, n*num_batch+i)
                print("step:%d,  loss:%g" % (n*num_batch+i, loss))

                if (n*num_batch+i) % 100 == 0:
                    acc = accuracy.eval({x: test_x, y_: test_y, keep_prob_fifty: 1.0, keep_prob_seventy_five: 1.0})
                    print("step:%d,  acc:%g" % (n*num_batch+i, acc))
                    """训练到准确率达到98%停止训练"""
                    if acc > 0.98 and n > 2:
                        saver.save(sess, './train_faces.model', global_step=n*num_batch+i)
                        sys.exit(0)


if __name__ == '__main__':

    """1、读取人脸数据"""
    readData(faces_my_path)
    readData(faces_other_path)
    imgs = np.array(imgs)                   # 将图片数据与标签转换成数组
    labs = np.array([[0, 1] if lab == faces_my_path else [1, 0] for lab in labs])  # 标签：[0,1]表示是我的人脸，[1,0]表示其他的人脸
    """2、随机划分测试集与训练集"""
    train_x_1, test_x_1, train_y, test_y = train_test_split(imgs, labs, test_size=0.05, random_state=random.randint(0, 100))
    train_x_2 = train_x_1.reshape(train_x_1.shape[0], size, size, 3)        # 参数：图片数据的总数，图片的高、宽、通道
    test_x_2 = test_x_1.reshape(test_x_1.shape[0], size, size, 3)
    """3、归一化"""
    train_x = train_x_2.astype('float32')/255.0
    test_x = test_x_2.astype('float32')/255.0
    print('Train Size:%s, Test Size:%s' % (len(train_x), len(test_x)))

    num_batch = len(train_x) // batch_size
    x = tf.placeholder(tf.float32, [None, size, size, 3])                 # 输入X：64*64*3
    y_ = tf.placeholder(tf.float32, [None, 2])                            # 输出Y_：1*2
    keep_prob_fifty = tf.placeholder(tf.float32)                          # 50%，即0.5
    keep_prob_seventy_five = tf.placeholder(tf.float32)                   # 75%，即0.75
    """4、进行训练"""
    train()

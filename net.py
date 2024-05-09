"""-----------------------------------------
net定义了神经网络结构
-----------------------------------------"""

import tensorflow as tf

"""定义神经网络层，共八层，卷积层特征提取，池化层降维,全连接层进行分类，共四类"""
def layer_net(input_image, num_class, dropout_rate, dropout_rate_2):
    """第一、二层，输入图片64*64*3，输出图片32*32*32"""
    w1 = tf.Variable(tf.random_normal([3, 3, 3, 32]))                 # 卷积核大小(3,3)， 输入通道(3)， 输出通道(32)
    b1 = tf.Variable(tf.random_normal([32]))
    layer_conv1 = tf.nn.relu(tf.nn.conv2d(input_image, w1, strides=[1, 1, 1, 1], padding='SAME')+b1)    # 64*64*32，卷积提取特征，增加通道数
    layer_pool1 = tf.nn.max_pool(layer_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 32*32*32，池化降维，减小复杂度
    drop1 = tf.nn.dropout(layer_pool1, dropout_rate)      # 按一定概率随机丢弃一些神经元，以获得更高的训练速度以及防止过拟合

    """第三、四层，输入图片32*32*32，输出图片16*16*64"""
    w2 = tf.Variable(tf.random_normal([3, 3, 32, 64]))  # 卷积核大小(3,3)， 输入通道(32)， 输出通道(64)
    b2 = tf.Variable(tf.random_normal([64]))
    layer_conv2 = tf.nn.relu(tf.nn.conv2d(drop1, w2, strides=[1, 1, 1, 1], padding='SAME') + b2)        # 32*32*64
    layer_pool2 = tf.nn.max_pool(layer_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')       # 16*16*64
    drop2 = tf.nn.dropout(layer_pool2, dropout_rate)

    """第五、六层，输入图片16*16*64，输出图片8*8*64"""
    w3 = tf.Variable(tf.random_normal([3, 3, 64, 64]))  # 卷积核大小(3,3)， 输入通道(64)， 输出通道(64)
    b3 = tf.Variable(tf.random_normal([64]))
    layer_conv3 = tf.nn.relu(tf.nn.conv2d(drop2, w3, strides=[1, 1, 1, 1], padding='SAME') + b3)        # 16*16*64
    layer_pool3 = tf.nn.max_pool(layer_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')       # 8*8*64=4096
    drop3 = tf.nn.dropout(layer_pool3, dropout_rate)


    """第七层，全连接层，将图片的卷积输出压扁成一个一维向量，输入图片8*8*64，reshape到1*4096，输出1*512"""
    w4 = tf.Variable(tf.random_normal([8*8*64, 512]))     # 输入通道(4096)， 输出通道(512)
    b4 = tf.Variable(tf.random_normal([512]))
    layer_fully_connected = tf.reshape(drop3, [-1, 8*8*64])         # -1表示行随着列的需求改变，1*4096
    relu = tf.nn.relu(tf.matmul(layer_fully_connected, w4) + b4)   # [1,4096]*[4096,512]=[1,512]
    drop4 = tf.nn.dropout(relu, dropout_rate_2)

    """第八层，输出层，输入1*512，输出1*2，再add"""
    w5 = tf.Variable(tf.random_normal([512, num_class]))        # 输入通道(512)， 输出通道(2)
    b5 = tf.Variable(tf.random_normal([num_class]))
    outdata = tf.add(tf.matmul(drop4, w5), b5)     # (1,512)*(512,2)=(1,2) ,跟input_label [0,1]、[1,0]比较给出损失 ，先乘再加
    return outdata
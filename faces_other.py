"""---------------------------------------------------------------
二、采集其他人脸数据集
有耶鲁大学的Yale人脸库、剑桥大学的ORL人脸库、美国国防部的FERET人脸库等
本系统使用人脸数据集下载:http://vis-www.cs.umass.edu/lfw/lfw.tgz
先将下载的图片集放在img_source目录下，用dlib来批量识别图片中的人脸部分，
并保存到指定目录faces_other
人脸大小：64*64
----------------------------------------------------------------"""
# -*- codeing: utf-8 -*-
import sys
import cv2
import os
import dlib

source_path = './img_source'
faces_other_path = './faces_other2'
size = 64
if not os.path.exists(faces_other_path):
    os.makedirs(faces_other_path)

"""特征提取器:dlib自带的frontal_face_detector"""
detector = dlib.get_frontal_face_detector()

num = 1
"""其中./path/dirnames/filenames"""
for (path, dirnames, filenames) in os.walk(source_path):
    for filename in filenames:
        if filename.endswith('.jpg'):
            print('time %s' % num)
            img_path = path+'/'+filename
            img = cv2.imread(img_path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            """ 使用detector进行人脸检测 dets为返回的结果"""
            dets = detector(gray_img, 1)

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

                face = img[x1:y1,x2:y2]
                face = cv2.resize(face, (size,size))   # 调整图片的尺寸
                cv2.imshow('image',face)
                cv2.imwrite(faces_other_path+'/'+str(num)+'.jpg', face)   #保存
                num += 1

            key = cv2.waitKey(30)
            if key == 27:
                sys.exit(0)

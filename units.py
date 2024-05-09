"""-----------------------------------------
units用到的一些模块
-----------------------------------------"""

import cv2


"""将loss写入txt文件中"""
def write_txt(txt_path, object_all):
    t = ''
    if len(object_all) == 0:
        pass
    with open(txt_path, 'w') as f:
        for line in object_all:
            for a in line:
                t = t + str(a)
            f.writelines(t)
            f.writelines('\n')
            t = ''

"""定义尺寸变换函数"""
def img_padding(img):
    height, width, _ = img.shape
    width_max = max(height, width)
    t, b, l, r = (0, 0, 0, 0)
    if width < width_max:
        tmp = width_max - width
        l = tmp // 2
        r = tmp - l
    elif height < width_max:
        tmp = width_max - height
        t = tmp // 2
        b = tmp - t
    else:
        pass
    return t, b, l, r
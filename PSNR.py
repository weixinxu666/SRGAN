from PIL import Image
import numpy
import math
import cv2
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
#导入你要测试的图像
im = numpy.array (Image.open ('D:\chrome download\SRCNN-keras-master\\0000.png'),'f')#将图像1数据转换为float型
im = cv2.resize(im, (200, 200), interpolation=cv2.INTER_CUBIC)
im2 = numpy.array (Image.open ('D:\chrome download\srgan-master\srgan-master\samples\evaluate\\valid_lr.png'),'f')#将图像2数据转换为float型
im2= cv2.resize(im2, (200, 200), interpolation=cv2.INTER_CUBIC)
print (im.shape,im.dtype)
#图像的行数
height = im.shape[0]
#图像的列数
width = im.shape[1]

def to_gray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#提取R通道
r = im[:,:,0]
#提取g通道
g = im[:,:,1]
#提取b通道
b = im[:,:,2]
#打印g通道数组
#print (g)
#图像1,2各自分量相减，然后做平方；
R = im2[:,:,0]-im[:,:,0]
G = im2[:,:,1]-im[:,:,1]
B = im2[:,:,2]-im[:,:,2]
#做平方
mser = R*R
mseg = G*G
mseb = B*B
#三个分量差的平方求和
SUM = mser.sum() + mseg.sum() + mseb.sum()
MSE = SUM / (height * width * 3)
PSNR = 10*math.log ( (255.0*255.0/(MSE)) ,10)

print (PSNR)
# im = numpy.array (Image.open ('D:\chrome download\srgan-master\srgan-master\samples\evaluate\\valid_gen.png'))#无符号型
# im2 = numpy.array (Image.open ('D:\chrome download\srgan-master\srgan-master\samples\evaluate\\valid_lr.png'))
# print("psnr",compare_ssim(im2,im))
print("ssim",compare_ssim(to_gray(im2),to_gray(im)))
# plt.subplot (121)#窗口1
# plt.title('origin image')
# plt.imshow(im,plt.cm.gray)
#
# plt.subplot(122)#窗口2
# plt.title('rebuilt image')
# plt.imshow(im2,plt.cm.gray)
# plt.show()


# PSNR.py

import numpy as np
import math

#
# def psnr(target, ref, scale):
#     # target:目标图像  ref:参考图像  scale:尺寸大小
#     # assume RGB image
#     target_data = np.array(target)
#     target_data = target_data[scale:-scale, scale:-scale]
#
#     ref_data = np.array(ref)
#     ref_data = ref_data[scale:-scale, scale:-scale]
#
#     diff = ref_data - target_data
#     diff = diff.flatten('C')
#     rmse = math.sqrt(np.mean(diff ** 2.))


# '''
# compute PSNR with tensorflow
# '''
# import tensorflow as tf
#
#
# def read_img(path):
#     return tf.image.decode_image(tf.read_file(path))
#
#
# def psnr(tf_img1, tf_img2):
#     return tf.image.psnr(tf_img1, tf_img2, max_val=255)
#
#
# def _main():
#     # t1 = tf.gfile.GFile('D:\chrome download\srgan-master\srgan-master\samples\evaluate\\valid_lr.png', 'r').read()
#     t1 = tf.image.decode_png('D:\chrome download\srgan-master\srgan-master\samples\evaluate\\valid_lr.png')
#     t1 = tf.image.resize_images(t1,[400,400])
#     # t2 = tf.gfile.GFile('D:\chrome download\srgan-master\srgan-master\samples\evaluate\\valid_gen.png', 'r').read()
#     t2 = tf.image.decode_png('D:\chrome download\srgan-master\srgan-master\samples\evaluate\\valid_gen.png')
#     t2 = tf.image.resize_images(t2,[400,400])
#
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         y = sess.run(psnr(t1, t2))
#         print(y)
# #
# if __name__ == '__main__':
#     _main()
#
# # Read images from file.
# im1 = tf.image.decode_png('D:\chrome download\srgan-master\srgan-master\samples\evaluate\\valid_lr.png')
# im2 = tf.image.decode_png('D:\chrome download\srgan-master\srgan-master\samples\evaluate\\valid_gen.png')
# # Compute PSNR over tf.uint8 Tensors.
# psnr1 = tf.image.psnr(im1, im2, max_val=255)
# # print(type(psnr1))
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     y = sess.run(psnr1)
#     print(y)
#
# # Compute PSNR over tf.float32 Tensors.
# im1 = tf.image.convert_image_dtype(im1, tf.float32)
# # print(im1)
# im2 = tf.image.convert_image_dtype(im2, tf.float32)
# psnr2 = tf.image.psnr(im1, im2, max_val=1.0)
# # psnr1 and psnr2 both have type tf.float32 and are almost equal.



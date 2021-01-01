# Author: Luca Onesto

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

def print_img(img,msg):
    print(msg)
    print('Type Enter to Continue')
    cv2.imshow('Image',img)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

def img_resize(img,resize_factor):
    if resize_factor <= 0 or resize_factor >= 100:
        return(img)
    else:
        width = int(img.shape[1] * (100-resize_factor) / 100)
        height = int(img.shape[0] *(100-resize_factor) / 100)
        dim = (width, height)
        # resize image
        resized_img = cv2.resize(img, dim)
        return(resized_img)


#import image through opencv
gray_img = cv2.imread('screen.png',cv2.IMREAD_GRAYSCALE)
# print_img(gray_img,'Plot grayscale image')

#edge detection
canny_th = 120
edges_img = cv2.Canny(gray_img,canny_th,canny_th)

# plot images in parallel
result_img = np.hstack((gray_img, edges_img))
title_image = 'Resulting_img '+str(canny_th)+'.png'
cv2.imwrite(title_image,result_img)

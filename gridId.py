# Author: Luca Onesto

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

def print_img(img,msg):
    print(msg)
    cv2.imshow('Image',img)
    cv2.waitKey()
    cv2.destroyAllWindows()

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
img = cv2.imread('screen.png')

# print_img(img,'Showing original image..')

#resize image
resized_img = img_resize(img,40)
# print_img(resized_img,'Showing resized image..')

#grayscale image
gray_img = cv2.cvtColor(resized_img,cv2.COLOR_BGR2GRAY)
# print_img(gray_img,'Grayscale image..')

#edge detection
edges_img = cv2.Canny(gray_img,30,100)
print_img(edges_img,'Edges image..Type Enter to Continue')

#hough transmorm
lines = cv2.HoughLines(edges_img,1,np.pi/18,150)

line_color = (0,255,0)

# Draw the lines
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(resized_img, pt1, pt2, line_color)

#printing and saving the last image
imgplot = plt.imshow(resized_img)
plt.show()
cv2.imwrite('detected_grid.png',resized_img)

import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

array_of_img = [] # this if for store all of the image data
# this function is for read image,the input is directory name
def read_directory(directory_name):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(r"./"+directory_name):
        #print(filename) #just for test
        #img is used to store the image data 
        img = cv2.imread(directory_name + "/" + filename)
        array_of_img.append(img)

def Stereo_Disparity_Map():
    read_directory("Q3_Image")
    imgL = array_of_img[0]
    imgR = array_of_img[1]
    gray_imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    gray_imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(gray_imgL, gray_imgR)
    plt.imshow(disparity, 'gray')
    plt.show()
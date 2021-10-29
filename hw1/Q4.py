import sys
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt

array_of_img = [] # this if for store all of the image data
# this function is for read image,the input is directory name
def read_directory(directory_name):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(r"./"+directory_name):
        #print(filename) #just for test
        #img is used to store the image data 
        img = cv2.imread(directory_name + "/" + filename)
        array_of_img.append(img)


def keypoint():
    read_directory("Q4_Image")
    img1=cv2.imread(r'./Q4_Image/Shark1.jpg')
    img2=cv2.imread(r'./Q4_Image/Shark2.jpg')

    #sift
    sift = cv2.xfeatures2d.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

    #feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1,descriptors_2)
    matches = sorted(matches, key = lambda x:x.distance)

    img1_idx = []
    img2_idx = []
    for i in range(0,200):
        img1_idx.append(keypoints_1[matches[i].queryIdx])
        img2_idx.append(keypoints_2[matches[i].trainIdx])

        #draw keypoint
    imgN1 = cv2.drawKeypoints(img1, keypoints=img1_idx, outImage=img1, color= (0, 0, 255), flags= cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    imgN2 = cv2.drawKeypoints(img2, keypoints=img2_idx, outImage=img2, color= (0, 0, 255), flags= cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    numpy_horizontal_concat = np.concatenate((imgN1, imgN2), axis=1)
    cv2.imshow('img', numpy_horizontal_concat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def matched_keypoint():
    read_directory("Q4_Image")
    img2=cv2.imread(r'./Q4_Image/Shark1.jpg')
    img1=cv2.imread(r'./Q4_Image/Shark2.jpg')

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #sift
    sift = cv2.xfeatures2d.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

    #feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1,descriptors_2)
    matches = sorted(matches, key = lambda x:x.distance)

    img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
    cv2.imshow('img',img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def wrap_image():
    read_directory("Q4_Image")

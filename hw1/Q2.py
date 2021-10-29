import sys
import cv2
import numpy as np
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

def shift_line(line,shift):
    for i in range(3):
        line[0][i]+=shift[i]
        line[1][i]+=shift[i]
    return line

def show_word(str):
    read_directory("Q2_Image")
    fs = cv2.FileStorage(r'./Q2_lib/alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(10,7,0)
    objp = np.zeros((11*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    for i in range(len(array_of_img)) :
        ret,corners=cv2.findChessboardCorners(array_of_img[i],(11,8),None)
        if ret==True:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, array_of_img[0].shape[::-2], None, None)
    for i in range(len(array_of_img)) :
        for j in range(len(str)) :
            ch=fs.getNode(str[j]).mat()
            for line in ch:
                line=shift_line(line,[7-j%3*3,5-int(j/3)*3,0])
                line=np.float32(line).reshape(-1,3)
                img_line, jac = cv2.projectPoints(line, rvecs[i], tvecs[i], mtx, dist)
                pt1=tuple(map(int,img_line[0].ravel()))
                pt2=tuple(map(int,img_line[1].ravel()))
                array_of_img[i]=cv2.line(array_of_img[i],pt1,pt2,(0,0,255),5)
        array_of_img[i]=cv2.resize(array_of_img[i], (720, 720))
        cv2.imshow('img',array_of_img[i])
        cv2.waitKey(1000)
    cv2.destroyAllWindows()


def show_vertically(str):
    read_directory("Q2_Image")
    fs = cv2.FileStorage(r'./Q2_lib/alphabet_lib_vertical.txt', cv2.FILE_STORAGE_READ)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(10,7,0)
    objp = np.zeros((11*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    for i in range(len(array_of_img)) :
        ret,corners=cv2.findChessboardCorners(array_of_img[i],(11,8),None)
        if ret==True:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, array_of_img[0].shape[::-2], None, None)
    for i in range(len(array_of_img)) :
        for j in range(len(str)) :
            ch=fs.getNode(str[j]).mat()
            for line in ch:
                line=shift_line(line,[7-j%3*3,5-int(j/3)*3,0])
                line=np.float32(line).reshape(-1,3)
                img_line, jac = cv2.projectPoints(line, rvecs[i], tvecs[i], mtx, dist)
                pt1=tuple(map(int,img_line[0].ravel()))
                pt2=tuple(map(int,img_line[1].ravel()))
                array_of_img[i]=cv2.line(array_of_img[i],pt1,pt2,(0,0,255),5)
        array_of_img[i]=cv2.resize(array_of_img[i], (720, 720))
        cv2.imshow('img',array_of_img[i])
        cv2.waitKey(1000)
    cv2.destroyAllWindows()
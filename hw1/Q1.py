import sys
import os
import numpy as np
import cv2 as cv
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import ui

array_of_img = [] # this if for store all of the image data
# this function is for read image,the input is directory name
def read_directory(directory_name):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(r"./"+directory_name):
        #print(filename) #just for test
        #img is used to store the image data 
        img = cv2.imread(directory_name + "/" + filename)
        array_of_img.append(img)


def find_corner():
    read_directory("Q1_Image")
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((8*11,3), np.float32)
    objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    for i in range(len(array_of_img)) :
        img = array_of_img[i]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (11,8), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (11,8), corners2, ret)
            img=cv2.resize(img,(720,720))
            cv2.imshow('img', img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()

def find_intrinsic():
    read_directory("Q1_Image")
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((8*11,3), np.float32)
    objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    for i in range(len(array_of_img)) :
        img = array_of_img[i]
        ret,corners=cv.findChessboardCorners(img,(11,8),None)
        if ret==True:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-2], None, None)
    print("Intrinsic Matrix:")
    print(mtx)

def find_distortion():
    read_directory("Q1_Image")
    objp = np.zeros((8*11,3), np.float32)
    objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    for i in range(len(array_of_img)) :
        img = array_of_img[i]
        ret,corners=cv2.findChessboardCorners(img,(11,8),None)
        if ret==True:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img.shape[::-2], None, None)
    print("Distortion Matrix:")
    print(dist)
    

def show_result():
    read_directory("Q1_Image")
    objp = np.zeros((8*11,3), np.float32)
    objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    for i in range(len(array_of_img)) :
        ret,corners=cv2.findChessboardCorners(array_of_img[i],(11,8),None)
        if ret==True:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, array_of_img[0].shape[::-2], None, None)
    for i in range(len(array_of_img)) :
        h, w=array_of_img[i].shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        # undistort
        dst = cv2.undistort(array_of_img[i], mtx, dist, None, newcameramtx)
        dst=np.append(dst,array_of_img[i],axis=1)#dst = dst[y:y+h,x:x+w] 
        dst=cv2.resize(dst,(1440,720))
        # crop the image
        #x, y, w, h = roi
        cv2.imshow('img', dst)
        cv2.waitKey(500)
    cv2.destroyAllWindows()

def find_extrinsic(image):
    read_directory("Q1_Image")
    image_target=int(image)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((8*11,3), np.float32)
    objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    for i in range(len(array_of_img)) :
        img = array_of_img[i]
        ret,corners=cv2.findChessboardCorners(img,(11,8),None)
        if ret==True:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-2], None, None)
    rvecs = cv2.Rodrigues(rvecs[image_target-1])
    print("Extrinsic Matrix:")
    print(np.append(rvecs[0],tvecs[image_target-1],axis=1))
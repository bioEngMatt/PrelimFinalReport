import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (9,4)
frameSize = (3280,2464)
LOAD_POINTS = True

RAND_SAMP = 200

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

objp = objp * 28.7
#print(objp)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.


imagesLeft = sorted(glob.glob('images/stereoLeft/*.png'))
imagesRight = sorted(glob.glob('images/stereoRight/*.png'))

sizes =  716
vidLeft = "./CalibrationCaptures/0_%s_sampVidLeftFull_tst.avi"%sizes
vidRight = "./CalibrationCaptures/0_%s_sampVidRightFull_tst.avi"%sizes 

if not LOAD_POINTS:
    vidLeft = cv.VideoCapture(vidLeft)
    vidRight = cv.VideoCapture(vidRight)
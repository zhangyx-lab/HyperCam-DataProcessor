#!python3
# System Packages
import sys
from os.path import exists
# Environment configurations
import env
# PIP Packages
import numpy as np
import cv2 as cv
# User packages
from util.util import rdGray
# global variable
CAM_MTX_PATH = env.VAR_PATH / "CAM_MTX.npy"
mtx = np.load(CAM_MTX_PATH) if exists(CAM_MTX_PATH) else None
CAM_DIST_PATH = env.VAR_PATH / "CAM_DIST.npy"
dist = np.load(CAM_DIST_PATH) if exists(CAM_DIST_PATH) else None


def init(SHOW_WINDOW=False):
    """Function to initialize undistort parameters"""
    print("Initializing undistortion parameters ...")
    # Checkerboard constants
    GRID = (6, 6)
    H, W, D = 0, 0, 0
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((GRID[0] * GRID[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:GRID[0], 0:GRID[1]].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    img = None
    # Initialize window thread if necessary
    if SHOW_WINDOW:
        WINDOW_NAME = "Original Image"
        cv.namedWindow(WINDOW_NAME, cv.WINDOW_AUTOSIZE)
        cv.startWindowThread()
    for path in env.CALIB_CHECKER_LIST:
        img = rdGray(path)
        H, W = img.shape
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(img, GRID, None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(
                img, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            if SHOW_WINDOW:
                cv.drawChessboardCorners(img, GRID, corners2, ret)
                cv.imshow(WINDOW_NAME, img)
                cv.setWindowProperty(WINDOW_NAME, cv.WND_PROP_TOPMOST, 1)
                cv.waitKey(10)
        else:
            print("  ERROR - No checkerboard found in {}".format(path),
                  file=sys.stderr)
    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, img.shape[::-1], None, None
    )
    np.save(CAM_MTX_PATH, mtx)
    np.save(CAM_DIST_PATH, dist)
    return mtx, dist


def apply(img, mtx=mtx, dist=dist):
    # Validate arguments
    if (mtx is None or dist is None):
        print("Missing input arguments:", file=sys.stderr)
        print("  img  {}".format(type(img)), file=sys.stderr)
        print("  mtx  {}".format(type(mtx)), file=sys.stderr)
        print("  dist {}".format(type(dist)), file=sys.stderr)
        raise RuntimeError()
    # Apply undistort to input image
    result = cv.undistort(img, mtx, dist)
    # crop and return the image
    h, w = result.shape
    dX = 160  # magic
    return result[:, dX:h+dX]

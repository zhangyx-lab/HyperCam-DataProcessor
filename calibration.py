#!python3
import numpy as np
import cv2 as cv
import glob
import sys
import re
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
images = list(glob.glob('checker/*.png'))
WINDOW_NAME = "Original Image"
cv.namedWindow(WINDOW_NAME, cv.WINDOW_AUTOSIZE)
cv.startWindowThread()
for fname in images:
    print("Processing %s" % fname)
    img = cv.cvtColor(cv.imread(fname), cv.COLOR_BGR2GRAY)
    H, W = img.shape
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(img, GRID, None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, GRID, corners2, ret)
        cv.imshow(WINDOW_NAME, img)
        cv.setWindowProperty(WINDOW_NAME, cv.WND_PROP_TOPMOST, 1)
        cv.waitKey(10)
    else:
        print(" - No checkerboard found")
# Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, img.shape[::-1], None, None
)
print('W =', W, ', H =', H, ', D =', D)
print(mtx)
print(dist)
# Prepare un-distort core
newCamMtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (H, W), 1, (H, W))
dX = 160
# Prepare brightness map
images = list(glob.glob('ref_white/*.png'))
bri_map = {}
for fname in images:
    ID = fname.replace("ref_white/", "").replace(".png", "")
    img = cv.cvtColor(cv.imread(fname), cv.COLOR_BGR2GRAY)
    # undistort
    dst = cv.undistort(img, mtx, dist)
    # crop the image according to AOI
    h, w = dst.shape
    dst = dst[:, dX:h+dX].astype(np.float32)
    dst = (np.ones(dst.shape) * 255).astype(np.float32) / dst
    bri_map[ID] = dst
# Show corrected checker
images = list(glob.glob('checker/*.png'))
WINDOW_NAME = "Corrected Checker"
cv.namedWindow(WINDOW_NAME, cv.WINDOW_AUTOSIZE)
cv.startWindowThread()
for fname in images:
    img = cv.cvtColor(cv.imread(fname), cv.COLOR_BGR2GRAY)
    result = re.findall("(?<=checker/)\w+(?=\.png)", fname)
    if len(result) == 0:
        print("Error: unable to extract color info from file name '{}'".format(
            fname), file=sys.stderr)
        continue
    img_ID = result[0]
    # undistort
    dst = cv.undistort(img, mtx, dist)
    # crop the image according to AOI
    h, w = dst.shape
    _dst = dst[:, dX:h+dX]
    dst = _dst.astype(np.float32) * bri_map[img_ID]
    dst = np.minimum(dst, np.ones(dst.shape) * 255).astype(np.uint8)
    print(np.max(dst), np.min(dst))
    # Show result
    BAR = np.zeros((h, 20))
    cv.imshow(WINDOW_NAME, np.concatenate(
        [img[0:h, dX:h+dX], BAR, _dst, BAR, dst], axis=1).astype(np.uint8))
    cv.setWindowProperty(WINDOW_NAME, cv.WND_PROP_TOPMOST, 1)
    if (cv.waitKey(0) != 13):
        sys.exit(0)
    break
# Show corrected images
WINDOW_NAME = "Corrected Data"
cv.namedWindow(WINDOW_NAME, cv.WINDOW_AUTOSIZE)
cv.startWindowThread()
images = list(glob.glob('data/*.png'))
for fname in images:
    print("Processing %s" % fname)
    img = cv.cvtColor(cv.imread(fname), cv.COLOR_BGR2GRAY)
    result = re.findall("(?<=_)\w+(?=\.png)", fname)
    if len(result) == 0:
        print("Error: unable to extract color info from file name '{}'".format(
            fname), file=sys.stderr)
        continue
    img_ID = result[0]
    # undistort
    dst = cv.undistort(img, mtx, dist)
    # crop the image according to AOI
    h, w = dst.shape
    dst = dst[:, dX:h+dX]
    dst = dst.astype(np.float32) * bri_map[img_ID]
    dst = np.minimum(dst, np.ones(dst.shape) * 255).astype(np.uint8)
    print(np.max(dst), np.min(dst))
    # Show result
    cv.imshow(WINDOW_NAME, dst)
    cv.setWindowProperty(WINDOW_NAME, cv.WND_PROP_TOPMOST, 1)
    cv.imwrite(fname.replace("data/", "results/"), dst)
    cv.waitKey(10)
# Wait until finished
# cv.destroyAllWindows()

#!python3
# System Packages
import sys
from os.path import exists
# PIP Packages
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# User packages
import env
from util.info import INFO, runtime_info, runtime_log
from util.util import rdGray
from util.convert import U8
from param import DTYPE, DTYPE_MAX
# global variable
CAM_MTX_PATH = env.CAL_CHECKER_PATH / "CAM_MTX.npy"
mtx = np.load(CAM_MTX_PATH) if exists(CAM_MTX_PATH) else None
CAM_DIST_PATH = env.CAL_CHECKER_PATH / "CAM_DIST.npy"
dist = np.load(CAM_DIST_PATH) if exists(CAM_DIST_PATH) else None
CAM_CROP_PATH = env.CAL_CHECKER_PATH / "CAM_CROP.npy"
crop = np.load(CAM_CROP_PATH) if exists(CAM_CROP_PATH) else None
# Parse from info.ini
INFO = INFO("Calibration.Checkerboard")
GRID = list(map(int, INFO("grid-count").split(",")))
CENTER_IMG = INFO("centered-grid")
CENTER_IMG_LED = INFO("centered-grid-led")
GRID_SIZE = INFO("size-per-grid", eval)
# Subpixel alignment termination criteria
CRITERIA = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def getCorners(img, grid=GRID, criteria=CRITERIA):
    if len(img.shape) > 2:
        img = img[:, :, 0]
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    # Convert data type
    img = U8(img)
    # Match corners
    ret, corners = cv.findChessboardCorners(img, grid, None)
    if ret != True:
        return None
    # Sub-pixel alignment
    return np.array(cv.cornerSubPix(
        img, corners,
        (11, 11), (-1, -1),
        criteria
    )).squeeze()


def getObjPoints(count: int, grid=GRID):
    """prepare object points, like (0,0,0), (1,0,0), (2,0,0), ...., (5,5,0)"""
    m, n = grid
    obj = np.zeros((m * n, 3), np.float32)
    obj[:, :2] = np.mgrid[:m, :n].T.reshape(-1, 2)
    return np.array([obj] * count)


def findCropBoundaries(mtx, dist):
    # Load image with the checkerbord in the center
    img = rdGray(env.CAL_CHECKER_PATH / CENTER_IMG)
    H, W = img.shape
    # Calculate undistorted points' coordinates
    d_points = getCorners(img)
    assert d_points is not None, f"Unable to find checkerboard in {CENTER_IMG}"
    u_points = cv.undistortPoints(d_points, mtx, dist, None, mtx)
    u_points = np.array(u_points).squeeze()
    # Get float coordinates of the cropping area
    crop = np.array([
        np.min(u_points, axis=0),
        np.max(u_points, axis=0)
    ])
    crop = np.maximum(crop, np.zeros(crop.shape))
    crop = np.minimum(crop, [[W, H]] * 2)
    # Crop parameters
    center = np.rint(np.average(crop, axis=0))
    limit = np.array([W, H])
    halfSize = np.rint(np.min(crop[1] - crop[0]) / 2)
    # Caculate Optimal New Size
    delta = int(np.min([halfSize, *center, *(limit - center)]))
    runtime_log(
        f"Cropping area is computed using:",
        f"  image with checker in cencter -> data/CAL_CHECKER/{CENTER_IMG}",
        f"  top right checker corner      -> {crop[1]}",
        f"  lower left checker corner     -> {crop[0]}",
    )
    runtime_info("cropping-area-size", 2 * delta)
    # Update cropping area coordinates
    crop = np.array([center - delta, center + delta], dtype=np.uint).T
    # Caculate pixel density according to delta-pixels
    actual_size = np.min(GRID) * GRID_SIZE  # mm
    pixel_density = delta / actual_size
    runtime_log(
        f"Pixel density is computed using:",
        f"  cropping area's actual size -> {actual_size:.4f} mm",
        f"  cropping area's pixel size  -> {delta} px",
    )
    runtime_info("pixel-density-our-camera", pixel_density)
    # Return the result
    return crop


def init(SHOW_WINDOW=False):
    """Function to initialize undistort parameters"""
    print("Initializing undistortion parameters ...")
    H, W, D = 0, 0, 0
    # Arrays to store object points and image points from all the images.
    img = None
    corner_points = []  # 2d points in image plane.
    # Initialize window thread if necessary
    if SHOW_WINDOW:
        WINDOW_NAME = "Original Image"
        cv.namedWindow(WINDOW_NAME, cv.WINDOW_AUTOSIZE)
        cv.startWindowThread()
    for path in env.CAL_CHECKER_LIST:
        img = rdGray(path)
        H, W = img.shape
        # Find the chess board corners
        corners = getCorners(img)
        # If found, add object points, image points (after refining them)
        if corners is not None:
            corner_points.append(corners)
            # Draw and display the corners
            if SHOW_WINDOW:
                cv.drawChessboardCorners(img, GRID, corners, ret)
                cv.imshow(WINDOW_NAME, img)
                cv.setWindowProperty(WINDOW_NAME, cv.WND_PROP_TOPMOST, 1)
                cv.waitKey(10)
        else:
            print(f"  - Warning: No checkerboard found in {path}")
    # Create 3D point mapping (Object Points)
    # 3d point in real world space
    object_points = getObjPoints(len(corner_points))
    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        object_points, corner_points, img.shape[::-1], None, None
    )
    # Find cropping area
    crop = findCropBoundaries(mtx, dist)
    # Save parameters
    np.save(CAM_MTX_PATH, mtx)
    np.save(CAM_DIST_PATH, dist)
    np.save(CAM_CROP_PATH, crop)
    return mtx, dist, crop


def demoUndistort(img, mtx=mtx, dist=dist, crop=crop):
    # Convert to uint8
    img = U8(img)
    # Convert to RGB
    if len(img.shape) < 3:
        img = np.stack([img[:, :]] * 3, axis=2)
    elif img.shape[2] < 3:
        img = np.stack([img[:, :, 0]] * 3, axis=2)
    # Find checker board corners
    corners = getCorners(img)
    assert corners is not None, "Failed to find corners in given image"
    # Draw original checker points
    COLOR = (255, 64, 64)
    # Detected corners
    dist_img = img.copy()
    for p in np.rint(corners).astype(np.int):
        cv.drawMarker(dist_img, p, COLOR, cv.MARKER_STAR, 40, 2)
    # Do transformation
    undist_img = apply(img, mtx, dist, None)
    # Draw corresponding points
    undist_corners = cv.undistortPoints(corners, mtx, dist, None, mtx)
    undist_corners = np.array(undist_corners).squeeze()
    # Cropping boundary
    cv.rectangle(undist_img, crop[:, 0], crop[:, 1], [0, 0, 255], 5)
    # Detected corners
    for p in np.rint(undist_corners).astype(np.int):
        cv.drawMarker(undist_img, p, COLOR, cv.MARKER_STAR, 40, 4)
    # Return result
    return dist_img, undist_img


def apply(img, mtx=mtx, dist=dist, crop=crop):
    # Validate arguments
    if (mtx is None or dist is None):
        print("Missing input arguments:", file=sys.stderr)
        print(f"  img  {type(img) }", file=sys.stderr)
        print(f"  mtx  {type(mtx) }", file=sys.stderr)
        print(f"  dist {type(dist)}", file=sys.stderr)
        raise RuntimeError()
    # Apply undistort to input image
    result = cv.undistort(img, mtx, dist)
    # Check of crop matrix was provided
    if crop is not None and crop.shape == (2, 2):
        # Load crop boundaries
        [x1, x2], [y1, y2] = crop
        # crop and return the image
        return result[y1:y2, x1:x2]
    else:
        # No crop
        return result


if __name__ == '__main__':
    init()

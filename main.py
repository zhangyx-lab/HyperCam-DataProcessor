#!python3
# Environment configurations
import env
from os import remove
from os.path import basename, exists
from glob import glob
from multiprocessing import get_context, cpu_count
# PIP Packages
import numpy as np
import cv2 as cv
from tqdm import tqdm
# User libraries
import util.whiteField as WhiteField
import util.undistort as Undistort
import util.refImage as RefImage
import util.align as Align
import util.util as util
import util.saveGrids as saveGrids
# Path constants
SAVE_PATH = env.CALIBRATED_PATH


def save_checker_sample(bri_map, mtx, dist):
    """Show corrected checker"""
    img_path = env.CALIB_CHECKER_LIST[0]
    img = util.rdGray(img_path)
    colorIndex = basename(img_path).replace('.png', '')
    # white field correction
    bri_corrected = WhiteField.apply(img, bri_map[colorIndex])
    # undistortion
    undistorted = Undistort.apply(bri_corrected, mtx=mtx, dist=dist)
    # output checkers to data path
    cv.imwrite(str(env.DATA_PATH / 'checker.raw.png'), img)
    cv.imwrite(str(env.DATA_PATH / 'checker.bri_corrected.png'), bri_corrected)
    cv.imwrite(str(env.DATA_PATH / 'checker.undistorted.png'), undistorted)


def run_calibrate(img_path):
    """Run image correction"""
    name = basename(img_path).replace('.png', '')

    # Read and convert raw image
    img = util.rdGray(img_path)
    # white field correction
    bri_corrected = WhiteField.apply(img, name)
    # undistortion
    undistorted = Undistort.apply(bri_corrected)
    # Save image
    cv.imwrite(str(SAVE_PATH / basename(img_path)), undistorted)


def get_gridView(id):
    saveGrids.apply(util.loadStack(id, env.RAW_PATH),
                    env.GRID_VIEW_PATH / "{}_raw.png".format(id))
    saveGrids.apply(util.loadStack(id, env.CALIBRATED_PATH),
                    env.GRID_VIEW_PATH / "{}_cal.png".format(id))

# Initiate cv2 window
# WINDOW_NAME = "Processed Image"
# cv.namedWindow(WINDOW_NAME, cv.WINDOW_AUTOSIZE)
# cv.startWindowThread()
# Run raw image calibration in parallel
def launch(fn, tasks):
    progress = pool.imap_unordered(fn, tasks)
    results = tqdm(progress, total=len(tasks), ascii=' >=')
    return [_ for _ in results]


if __name__ == '__main__':
    # Remove temporary files in var folder
    for f in list(glob(str(env.VAR_PATH / "*"))):
        remove(f)
    if exists(env.REPORT_PATH):
        remove(env.REPORT_PATH)
    env.REPORT_PATH.touch()
    # Initialize calibrations
    bri_map = WhiteField.init()
    mtx, dist = Undistort.init()
    # Check for calibration result
    save_checker_sample(bri_map, mtx, dist)
    # ----------------------
    with get_context("spawn").Pool(processes=cpu_count()) as pool:
        # Run calibration
        print("\nRunning calibration on raw images ...")
        launch(run_calibrate, env.RAW_IMAGES())
        # Gather ID list
        idList = util.getIdList(env.CALIBRATED_IMAGES())
        # Generate image grids
        print("\nGenerating gird views ...")
        launch(get_gridView, idList)
        # Prepare reference images
        print("\nInitializing reference images ...")
        launch(RefImage.init, env.REF_IMAGES())
        # Run image alignment
        print("\nAligning images to references ...")
        launch(Align.apply, idList)
    # Sort the report
    report = open(env.REPORT_PATH, 'r').readlines()
    report.sort()
    open(env.REPORT_PATH, 'w').writelines(report)

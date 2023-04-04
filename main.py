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
from util.info import INFO, runtime_info_init
from util.convert import U16, U8
import util.whiteField as WhiteField
import util.undistort as Undistort
import util.refImage as RefImage
import util.align as Align
import util.util as util
import util.saveGrids as SaveGrids
# Path constants
SAVE_PATH = env.CALIBRATED_PATH
# Info getter
getHyperCamInfo = INFO("RawImage.HyperCam")


def save_checker_sample():
    """Show corrected checker"""
    img_path = env.CAL_CHECKER_PATH / Undistort.CENTER_IMG
    img = util.rdGray(img_path)
    cv.imwrite(
        str(env.CAL_DEMO_PATH / '0.raw.png'),
        U8(img)
    )
    # Load color index from configuration
    colorIndex = Undistort.CENTER_IMG_LED
    # white field correction
    bri_corrected = WhiteField.apply(img, colorIndex)
    cv.imwrite(
        str(env.CAL_DEMO_PATH / '1.bri_correct.png'),
        U8(bri_corrected)
    )
    # undistortion demo
    corner, undist = Undistort.demoUndistort(bri_corrected)
    cv.imwrite(
        str(env.CAL_DEMO_PATH / '2.cornet_detect.png'),
        U8(corner)
    )
    cv.imwrite(
        str(env.CAL_DEMO_PATH / '3.undistort.png'),
        U8(undist)
    )
    # undistortion result
    undistorted = Undistort.apply(bri_corrected)
    # output checkers to data path
    cv.imwrite(
        str(env.CAL_DEMO_PATH / '4.result.png'),
        U8(undistorted)
    )


def run_calibrate(img_path):
    """Run image correction"""
    name = basename(img_path).replace('.png', '')
    # Read and convert raw image
    img = util.rdGray(img_path)
    # white field correction
    bri_corrected = WhiteField.apply(img, name)
    # undistortion
    undistorted = Undistort.apply(bri_corrected)
    # Check for image rotation
    if getHyperCamInfo("rotation", int) == 180:
        undistorted = undistorted[::-1, ::-1]
    # Save image
    cv.imwrite(str(SAVE_PATH / basename(img_path)), U16(undistorted))


def get_gridView(id):
    stack = util.loadStack(id, env.RAW_PATH)
    SaveGrids.apply(stack, f"{id}_Raw.png")
    stack = util.loadStack(id, env.CALIBRATED_PATH)
    SaveGrids.apply(stack, f"{id}_Cal.png")
    # Save as numpy
    np.save(env.CALIBRATED_PATH / id, stack)
    # Save kernel
    cv.imwrite(str(env.ALIGN_KERNEL_PATH /
               f"{id}.png"), Align.getKernel(stack))


if __name__ == '__main__':
    # Reload runtime info from template
    runtime_info_init()
    # Remove temporary files in var folder
    with open(env.REPORT_PATH, 'w') as r:
        r.write("")
    # Initialize calibrations
    WhiteField.init()
    Undistort.init()
    # Check for calibration result
    save_checker_sample()
    # ------------------------------------------------------------
    with get_context("spawn").Pool(processes=10) as pool:
        def launch(fn, tasks):
            """Run raw image calibration in parallel"""
            progress = pool.imap_unordered(fn, tasks)
            results = tqdm(progress, total=len(tasks), ascii=' >=')
            return [_ for _ in results]
        # Run calibration
        print("\nRunning calibration on raw images ...")
        launch(run_calibrate, env.RAW_IMAGES())
        # Gather ID list
        idList = util.getIdList(env.CALIBRATED_IMAGES())
        # Initialize alignment kernel size
        Align.init()
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

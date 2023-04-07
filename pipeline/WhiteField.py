#!python3
# Environment configurations
import env
# PIP Packages
import cvtb
import numpy as np
import cv2 as cv
from os.path import basename
# User package
import util.util as util


def init():
    """Function to initialize reference white map"""
    print("Initializing white field parameters ...")
    # Prepare brightness map
    for img_path in env.CAL_WHITE_LIST:
        colorIndex = basename(img_path).replace(".png", "")
        ref = cvtb.types.F32(util.rdGray(img_path))
        ref = np.maximum(ref, 1e-4)
        map = 1 / ref
        np.save(env.CAL_WHITE_PATH / colorIndex, cvtb.types.F32(map))


def load_map(colorIndex):
    return np.load(env.CAL_WHITE_PATH / f"{colorIndex}.npy")


def apply(img, white):
    if (isinstance(white, str)):
        white = load_map(util.getColorIndex(white))
    result = white * cvtb.types.F32(img)
    return result

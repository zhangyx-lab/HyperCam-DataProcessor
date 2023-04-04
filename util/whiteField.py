#!python3
# Environment configurations
import env
# PIP Packages
import numpy as np
import cv2 as cv
from os.path import basename
# User package
from param import DTYPE, DTYPE_MAX
import util.util as util
# global variable
bri_map = {}


def init():
    """Function to initialize reference white map"""
    print("Initializing white field parameters ...")
    # Prepare brightness map
    for img_path in env.CAL_WHITE_LIST:
        colorIndex = basename(img_path).replace(".png", "")
        ref = util.rdGray(img_path)
        map = (np.ones(ref.shape) * DTYPE_MAX).astype(np.float32) / ref
        bri_map[colorIndex] = map
        np.save(env.VAR_PATH / ("BRI_MAP_" + colorIndex + ".npy"), map)
    return bri_map


def load_map(colorIndex):
    return np.load(env.VAR_PATH / ("BRI_MAP_" + colorIndex + ".npy"))


def apply(img, map):
    if (isinstance(map, str)):
        map = load_map(util.getColorIndex(map))
    result = map * img
    result[result > DTYPE_MAX] = DTYPE_MAX
    return result.astype(DTYPE)

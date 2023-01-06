#!python3
# Environment configurations
import env
from os.path import basename
import warnings
# PIP Packages
import numpy as np
import cv2 as cv
import spectral.io.envi as envi


def load(name) -> np.ndarray:
    warnings.filterwarnings("ignore")
    path = str(env.REF_PATH / name)
    file = envi.open(path + ".hdr", path + ".dat")
    return np.array(file.load())


def init(path):
    """Function to initialize reference white map"""
    name = basename(path)
    data = load(name).astype(np.float32)[:, :, 100:199]
    (H, W, D) = data.shape
    for i in range(D):
        data[:, :, i] = data[:, :, i] / np.max(data[:, :, i])
    ref = (np.squeeze(np.average(data, axis=2)) * 255).astype(np.uint8)
    cv.imwrite(str(env.VAR_PATH / "REF_{}.png".format(name)), ref)

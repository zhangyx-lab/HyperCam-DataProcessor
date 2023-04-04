#!python3
# PIP Packages
import numpy as np
from numpy.typing import NDArray as NPA
import cv2 as cv
# User libraries
from util.convert import F32, trimToFit


def equalizeHist(cube: NPA, level: float):
    # cube = F32(cube)
    pass


def equalizeGeo(cube: NPA[np.float32], kernelSize: int) -> NPA[np.float32]:
    if kernelSize is None:
        return cube
    assert kernelSize % 2 == 1, "kernel size for equalization must be an ODD integer"
    ndim = cube.ndim
    assert ndim == 2 or ndim == 3, ndim
    # Convert cube to F32
    cube = F32(cube)
    # Flatten the image into grayscale
    flat = np.average(cube, axis=2) if ndim > 2 else cube.copy()
    # Compute the overall brightness channel
    level = np.average(flat)
    # Compute local brightness distribution
    local = cv.GaussianBlur(flat, [kernelSize] * 2, 0)
    # Compute the scale ration for each pixel
    scale = level / local
    scale = scale / np.average(scale)
    # Expand size of scale to fit cube's dimensions
    if cube.ndim == 3:
        scale = np.stack([scale] * cube.shape[2], axis=2)
    # Limit the value range within 0-1
    return trimToFit(cube * scale)

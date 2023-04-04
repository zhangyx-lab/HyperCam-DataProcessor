#!python3
# Environment configurations
import env
from os.path import basename
import warnings
# PIP Packages
import numpy as np
from numpy.typing import NDArray as NPA
import cv2 as cv
import spectral.io.envi as envi
# User libraries
from param import DTYPE, DTYPE_MAX
from util.info import INFO
from util.convert import REF2BGR, REF2GRAY, scaleToFit, trimToFit
getInfo = INFO("RawImage.Reference")

file = None
def load(name) -> NPA[np.float32]:
    global file
    warnings.filterwarnings("ignore")
    path = str(env.REF_PATH / name)
    file = envi.open(path + ".hdr", path + ".dat")
    cube = np.array(file.load())
    return cube


def equalize(cube: NPA[np.float32]) -> NPA[np.float32]:
    kernelSize = getInfo("intensity-equalizer-kernel", int, optional=True)
    if kernelSize is None: return cube
    assert kernelSize % 2 == 1, "kernel size for equalization must be an ODD integer"
    # Flatten the image into 1 channel
    flat = np.average(cube, axis=2)
    # Compute the overall brightness channel
    level = np.average(flat)
    # Compute local brightness distribution
    local = cv.GaussianBlur(flat, [kernelSize] * 2, 0)
    # Compute the scale ration for each pixel
    scale = level / local
    scale = scale / np.average(scale)
    # Return equalized cube
    equalized = np.stack([
        cube[:, :, b] * scale for b in range(cube.shape[2])
    ], axis=2)
    # Limit the value range within 0-1
    equalized[equalized < 0] = 0
    equalized[equalized > 1] = 1
    return equalized


def init(path):
    """Function to initialize reference image matching template"""
    name = basename(path)
    # Load from .hdr and .dat files
    cube = load(name)
    # Equalize brightness distribution (if configured)
    cube = equalize(cube)
    # Save as numpy
    np.save(env.VAR_PATH / f"REF_{name}", cube)
    # Generate fake color image
    cv.imwrite(
        str(env.VAR_PATH / f"REF_{name}.png"),
        REF2BGR(cube)
    )


if __name__ == '__main__':
    for p in env.REF_IMAGES():
        print(f"Processing {p}")
        init(p)

#!python3
# Environment configurations
import env
from os.path import basename
import warnings
# PIP Packages
import cvtb
import numpy as np
from numpy.typing import NDArray as NPA
import cv2 as cv
import spectral.io.envi as envi
# User libraries
from util.info import INFO
from util.convert import REF2BGR, REF2GRAY
getInfo = INFO("RawImage.Reference")
SAVE_PATH = env.REF_CAL_PATH
U8C1_PATH = env.ensureDir(SAVE_PATH / "U8C1")
kernelSize = getInfo("intensity-equalizer-kernel", int, True)


def load(name) -> NPA[np.float32]:
    warnings.filterwarnings("ignore")
    path = str(env.REF_PATH / name)
    file = envi.open(path + ".hdr", path + ".dat")
    cube = np.array(file.load())
    if getInfo("rotation", int) == 180:
        cube = cube[::-1, ::-1]
    return cube


def init(path):
    """Function to initialize reference image matching template"""
    name = basename(path)
    # Load from .hdr and .dat files
    cube = load(name)
    # Equalize brightness distribution (if configured)
    cube = cvtb.geometric.equalize(kernelSize)(cube)
    # Save as numpy
    np.save(SAVE_PATH / name, cube)
    # Generate fake color image
    cv.imwrite(
        str(SAVE_PATH / f"{name}.png"),
        cvtb.types.U8(REF2BGR(cube))
    )
    # Generate GrayScale NPY for matching
    gray = REF2GRAY(cube)
    gray *= 0.5 / np.average(gray)
    gray = cvtb.histogram.gamma(2)(gray)
    gray *= 0.5 / np.average(gray)
    np.save(U8C1_PATH / name, gray)
    cv.imwrite(
        str(U8C1_PATH / f"{name}.png"),
        cvtb.types.U8(gray)
    )


if __name__ == '__main__':
    for p in env.REF_IMAGES():
        print(f"Processing {p}")
        init(p)

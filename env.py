#!python3
from pathlib import Path
from glob import glob as glob
from os import mkdir, environ
from os.path import exists, dirname, realpath
# Debug level
if environ.get('DEBUG') is None:
    DEBUG = 0
else:
    DEBUG = int(environ.get('DEBUG') or 1)
    print(f"DEBUG LEVEL {DEBUG}")
# Base path of the project
BASE = Path(dirname(realpath(__file__)))
# Path constants
DATA_PATH        = BASE / "data"
# List of all dynamically written directorys
dynamic_paths    = []


def DATA_DIR(subdir: str, variable: bool = False):
    path = DATA_PATH / subdir
    if not exists(path):
        assert variable, f"Missing required data directory {path}"
        mkdir(path)
    if variable:
        dynamic_paths.append(path)
    return path
# Required data sources
RAW_PATH          = DATA_DIR("RAW")
REF_PATH          = DATA_DIR("REF")
CAL_CHECKER_PATH  = DATA_DIR("CAL_CHECKER")
CAL_WHITE_PATH    = DATA_DIR("CAL_WHITE")
# File that contains alignment reports
REPORT_PATH       = DATA_PATH / "report.txt"
# Generated file destinations
CAL_DEMO_PATH     = DATA_DIR("0-CalDemo"       , True)
CALIBRATED_PATH   = DATA_DIR("1-Calibrated"    , True)
GRID_VIEW_PATH    = DATA_DIR("2-GridView"      , True)
ALIGN_KERNEL_PATH = DATA_DIR("3-Kernels"       , True)
REF_CAL_PATH      = DATA_DIR("4-CalibratedRefs", True)
ALIGNED_PATH      = DATA_DIR("5-Aligned"       , True)


def ensureDir(path):
    """Create path if not exist"""
    if not exists(path):
        mkdir(path)
    return path


# Source image files
CAL_CHECKER_LIST = list(glob(str(CAL_CHECKER_PATH / "*.png")))
CAL_WHITE_LIST = list(glob(str(CAL_WHITE_PATH / "*.png")))
def RAW_IMAGES(): return list(glob(str(RAW_PATH / "*.png")))
def REF_IMAGES(): return [_.replace(".dat", "")
                          for _ in glob(str(REF_PATH / "*.dat"))]


def CALIBRATED_IMAGES(): return list(glob(str(CALIBRATED_PATH / "*.png")))


if __name__ == '__main__':
    for path in dynamic_paths:
        print(path)

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
VAR_PATH = BASE / "var"
DATA_PATH = BASE / "data"
# Required data sources
RAW_PATH = DATA_PATH / "RAW"
REF_PATH = DATA_PATH / "REF"
CAL_CHECKER_PATH = DATA_PATH / "CAL_CHECKER"
CAL_WHITE_PATH = DATA_PATH / "CAL_WHITE"
# Generated file destinations
REPORT_PATH = DATA_PATH / "report.txt"
CAL_DEMO_PATH = DATA_PATH / "0-CalDemo"
CALIBRATED_PATH = DATA_PATH / "1-Calibrated"
GRID_VIEW_PATH = DATA_PATH / "2-GridView"
ALIGNED_PATH = DATA_PATH / "3-Aligned"


def ensureDir(path):
    """Create path if not exist"""
    if not exists(path):
        mkdir(path)


for d in [
    VAR_PATH,
    CAL_DEMO_PATH, CALIBRATED_PATH,
    GRID_VIEW_PATH, ALIGNED_PATH
]:
    ensureDir(d)
# Source image files
CAL_CHECKER_LIST = list(glob(str(CAL_CHECKER_PATH / "*.png")))
CAL_WHITE_LIST = list(glob(str(CAL_WHITE_PATH / "*.png")))
def RAW_IMAGES(): return list(glob(str(RAW_PATH / "*.png")))
def REF_IMAGES(): return [_.replace(".dat", "")
                          for _ in glob(str(REF_PATH / "*.dat"))]


def CALIBRATED_IMAGES(): return list(glob(str(CALIBRATED_PATH / "*.png")))

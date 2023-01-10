from pathlib import Path
from glob import glob
from os import mkdir
from os.path import exists, dirname, realpath
# Size of the template matching kernel
KERNEL_SIZE = 286
# Color profiles for our camera
COLORS = [
    "Ultra_Violet",	# 0
    "Blue",			# 1
    "Green",		# 2
    "Yellow_Green",	# 3
    "Yellow",		# 4
    "Orange",		# 5
    "Red",			# 6
    "Infrared"		# 7
]
# Base path of the project
BASE = Path(dirname(realpath(__file__)))
# Path constants
VAR_PATH = BASE / "var"
DATA_PATH = BASE / "data"
RAW_PATH = DATA_PATH / "raw"
REF_PATH = DATA_PATH / "reference"
REPORT_PATH = DATA_PATH / "report.txt"
CALIBRATED_PATH = DATA_PATH / "calibrated"
ALIGNED_PATH = DATA_PATH / "aligned"
# Create paths if not exist
for path in [VAR_PATH, CALIBRATED_PATH, ALIGNED_PATH]:
    if not exists(path):
        mkdir(path)
# Source image files
CALIB_CHECKER_LIST = list(glob(str(BASE / "calib_checker" / "*.png")))
CALIB_WHITE_LIST = list(glob(str(BASE / "calib_white" / "*.png")))
RAW_IMAGES = list(glob(str(RAW_PATH / "*.png")))
REF_IMAGES = [_.replace(".dat", "") for _ in glob(str(REF_PATH / "*.dat"))]
CALIBRATED_IMAGES = list(glob(str(CALIBRATED_PATH / "*.png")))

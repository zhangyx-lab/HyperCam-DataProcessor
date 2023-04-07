from pathlib import Path
from os.path import basename
import cvtb
import re
import numpy as np
from numpy.typing import NDArray as NPA
from cv2 import imread, putText, LINE_AA, IMREAD_UNCHANGED
from cv2 import FONT_HERSHEY_DUPLEX as FONT
# Project Packages
import env
from param import COLORS


def loadStack(id: str, base=env.CALIBRATED_PATH) -> NPA[np.uint16]:
    stack = [f"{id}_{color}.png" for color in COLORS]
    stack = [rdGray(base / _) for _ in stack]
    return np.stack(stack, axis=2)


def rdGray(path: Path) -> NPA:
    img = imread(str(path), IMREAD_UNCHANGED)
    assert len(img.shape) == 2, img
    return img


def getIdList(path_list):
    names = [basename(s) for s in path_list]
    keys = [re.findall("^[A-Z]\d+(?=_)", s)[0] for s in names]
    return list(dict.fromkeys(keys))


def getColorIndex(file_name):
    result = re.findall("(?<=_)\w+$", file_name)
    if len(result) > 0:
        return result[0]
    else:
        return file_name

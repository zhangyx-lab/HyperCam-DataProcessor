from pathlib import Path
from os.path import basename
import re
from sys import stderr
from math import floor, ceil, log
from numpy import ndarray, squeeze, concatenate, average, ones, uint8, float32
from cv2 import resize as cv_resize, imread, cvtColor, COLOR_BGR2GRAY


def rdGray(path: Path) -> ndarray:
    img = imread(str(path))
    return cvtColor(img, COLOR_BGR2GRAY)


def gamma(img: ndarray, g: float = -1.0) -> ndarray:
    img = img.astype(float32) / 255
    if g <= 0.0:
        mean = average(img)
        g = log(0.5) / log(mean)
        # print("mean =", mean, "gamma =", g)
    # Run gamma correction
    img = (img ** g) * 255
    return img.astype(uint8)


def gammaAlign(img: ndarray, target: ndarray) -> ndarray:
    g = log(average(target) / 255.0) / log(average(img) / 255.0)
    return gamma(img, g)


def getIdList(path_list):
    names = [basename(s) for s in path_list]
    keys = [re.findall("^[A-Z]\d+(?=_)", s)[0] for s in names]
    return list(dict.fromkeys(keys))


def getColorIndex(file_name):
    result = re.findall("(?<=_)\w+$", file_name)
    if len(result) == 0:
        print("Error: unable to extract color info from file name '{}'".format(
            file_name), file=stderr)
        raise RuntimeError
    return result[0]


def pad(img: ndarray, color, h: int = 0, w: int = 0) -> ndarray:
    img = img.reshape((img.shape[0], img.shape[1], -1))
    img_h, img_w, img_d = img.shape
    # Fill in default values
    if h <= 0:
        h = img_h
    if w <= 0:
        w = img_w
    # Check if target size is larger than or equal to image size
    if img_h > h or img_w > w:
        print("Image ({} by {}) too large to fit into {} by {}".format(
            img_w, img_h, w, h), file=stderr)
        raise RuntimeError

    def gen(shape):
        """Padding generator"""
        return ones(shape, dtype=img.dtype) * color
    # Do padding
    dH, dW = h - img_h, w - img_w
    # Vertical padding
    shape = [(floor(dH / 2), img_w, img_d), (ceil(dH / 2), img_w, img_d)]
    img = concatenate((gen(shape[0]), img, gen(shape[1])), axis=0)
    # Horizontal padding
    shape = [(h, floor(dW / 2), img_d), (h, ceil(dW / 2), img_d)]
    img = concatenate((gen(shape[0]), img, gen(shape[1])), axis=1)
    return squeeze(img)


def resize(img: ndarray, h: int = 0, w: int = 0) -> ndarray:
    img = img.reshape((img.shape[0], img.shape[1], -1))
    img_h, img_w, img_d = img.shape
    # Validate arguments
    if h <= 0 and w > 0:
        h = round(w * img_h / img_w)
    elif w <= 0 and h > 0:
        w = round(h * img_w / img_h)
    else:
        print("Invalid target size: {} by {}".format(w, h), file=stderr)
        raise RuntimeError
    # Do resize
    return squeeze(cv_resize(img, (w, h)))

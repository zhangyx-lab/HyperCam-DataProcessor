from pathlib import Path
from os.path import basename
import re
from sys import stderr
from math import floor, ceil, log
from numpy import ndarray, squeeze, concatenate, average, ones, float32, absolute
import numpy as np
from numpy.typing import NDArray as NPA
from cv2 import imread, cvtColor, putText, COLOR_BGR2GRAY, LINE_AA, IMREAD_UNCHANGED
from cv2 import resize as cv_resize
from cv2 import FONT_HERSHEY_DUPLEX as FONT
# Project Packages
import env
from param import COLORS
import util.convert as cvt


def loadStack(id: str, base=env.CALIBRATED_PATH) -> NPA[np.uint16]:
    stack = ["{}_{}.png".format(id, _) for _ in COLORS]
    stack = [rdGray(base / _) for _ in stack]
    return np.stack(stack, axis=2)


def val_map(x, range) -> float:
    a, b = range
    if a > b:
        return 1 - val_map(x, (b, a))
    if x <= a:
        return float(0)
    if x >= b:
        return float(1)
    return (x - a) / (b - a)


def invert(rgb):
    return 255 - ndarray(rgb)


def wave2bgr(wave, invisible=0.3):
    def gamma(color, GAMMA):
        return (255 * pow(color, GAMMA)).astype(np.uint8)
    B = val_map(wave, (510, 490))
    G = np.min([val_map(wave, (440, 490)), val_map(wave, (645, 580))])
    R = np.max([val_map(wave, (440, 380)), val_map(wave, (510, 580))])
    intensity = np.max([
        np.min([
            val_map(wave, (380, 420)),
            val_map(wave, (780, 700))
        ]),
        invisible
    ])
    color = np.array([B, G, R], dtype=float32)
    return gamma(color * intensity, 0.8)


def draw_text(im, txt, pos=None, font=FONT, scale=0.5, color=[255] * 3, width=1):
    if pos is None:
        pos = (10, im.shape[0] - 10)
    return putText(im.astype(np.uint8), txt, pos, font, scale, color, width, LINE_AA)


def rdGray(path: Path) -> NPA:
    img = imread(str(path), IMREAD_UNCHANGED)
    assert len(img.shape) == 2, img
    return img


def contrast(img: NPA, c: float = 1.0):
    img = img.astype(float32)
    img = 2 * img / 255 - 1
    mask = img < 0
    img = absolute(img)
    img = img ** (1 / c)
    img = 255 * (img + 1) / 2
    img = img.astype(np.uint8)
    img[mask] = 255 - img[mask]
    return img


def gamma(img: NPA, g: float = -1.0):
    img = cvt.F32(img)
    if g <= 0:
        mean = average(img)
        g = log(0.5) / log(mean)
    # Run gamma correction
    img = (img ** g)
    return cvt.F32(img)


def gammaAlign(img: NPA, target: NPA):
    g = log(average(cvt.F32(target))) / log(average(cvt.F32(img)))
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


def pad(img: NPA, color=[255] * 3, h: int = 0, w: int = 0) -> NPA[np.uint8]:
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


def resize(img: NPA, h: int = 0, w: int = 0) -> ndarray:
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

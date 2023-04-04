from pathlib import Path
from os.path import basename
import re
from sys import stderr
from math import floor, ceil, log
from numpy import ndarray, squeeze, concatenate, average, ones, float32, absolute
import numpy as np
from cv2 import imread, cvtColor, putText, COLOR_BGR2GRAY, LINE_AA, IMREAD_UNCHANGED
from cv2 import resize as cv_resize
from cv2 import FONT_HERSHEY_DUPLEX as FONT
import env
from param import DTYPE, DTYPE_MAX, COLORS


def loadStack(id: str, base=env.CALIBRATED_PATH) -> np.ndarray:
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
    return DTYPE_MAX - ndarray(rgb)


def wave2bgr(wave, invisible=0.3):
    def gamma(color, GAMMA):
        return (DTYPE_MAX * pow(color, GAMMA)).astype(DTYPE)
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


def draw_text(im, txt, pos=None, font=FONT, scale=0.5, color=[DTYPE_MAX] * 3, width=1):
    if pos is None:
        pos = (10, im.shape[0] - 10)
    return putText(im.astype(DTYPE), txt, pos, font, scale, color, width, LINE_AA)


def rdGray(path: Path) -> ndarray:
    img = imread(str(path), IMREAD_UNCHANGED)
    assert len(img.shape) == 2, img
    return img


def contrast(img: ndarray, c: float = 1.0):
    img = img.astype(float32)
    img = 2 * img / DTYPE_MAX - 1
    mask = img < 0
    img = absolute(img)
    img = img ** (1 / c)
    img = DTYPE_MAX * (img + 1) / 2
    img = img.astype(DTYPE)
    img[mask] = DTYPE_MAX - img[mask]
    return img


def gamma(img: ndarray, g: float = -1.0) -> ndarray:
    flag_convert = img.dtype == DTYPE
    if flag_convert: img = img.astype(float32) / DTYPE_MAX
    if g <= 0.0:
        mean = average(img)
        g = log(0.5) / log(mean)
    # Run gamma correction
    img = (img ** g)
    if flag_convert:
        img = (img * DTYPE_MAX).astype(DTYPE)
    return img


def gammaAlign(img: ndarray, target: ndarray) -> ndarray:
    g = log(average(target) / DTYPE_MAX) / log(average(img) / DTYPE_MAX)
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


def pad(img: ndarray, color=[DTYPE_MAX] * 3, h: int = 0, w: int = 0) -> ndarray:
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

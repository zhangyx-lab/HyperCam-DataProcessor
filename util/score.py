import cv2 as cv
import numpy as np
from util.util import gamma


def score(src: np.ndarray, dst: np.ndarray) -> float:
    src = np.average(src, axis=2).astype(np.uint8)
    src = gamma(cv.equalizeHist(src)) >= 200
    dst = gamma(cv.equalizeHist(dst)) >= 200
    TT = np.logical_and(src, dst)
    R = np.logical_or(TT, src)
    B = np.logical_or(TT, dst)
    G = TT
    img = np.stack((B, G, R), axis=2).astype(np.uint8) * 255
    match_score = np.sum(TT) / np.sum(np.logical_or(src, dst))
    return match_score, img

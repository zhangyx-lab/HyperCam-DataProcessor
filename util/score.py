import cv2 as cv
import numpy as np
from util.util import gamma
from util.convert import U8
# User libraries
from param import DTYPE, DTYPE_MAX


def score(src: np.ndarray, dst: np.ndarray) -> float:
    src = gamma(cv.equalizeHist(U8(src))) >= 200
    dst = gamma(cv.equalizeHist(U8(dst))) >= 200
    TT = np.logical_and(src, dst)
    R = np.logical_or(TT, src)
    B = np.logical_or(TT, dst)
    G = TT
    img = np.stack((B, G, R), axis=2).astype(DTYPE) * DTYPE_MAX
    match_score = np.sum(TT) / np.sum(np.logical_or(src, dst))
    return match_score, img

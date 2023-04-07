import cvtb
import numpy as np


def score(src: np.ndarray, dst: np.ndarray) -> float:
    src = cvtb.histogram.gamma()(src) >= 0.5
    dst = cvtb.histogram.gamma()(dst) >= 0.5
    TT = np.logical_and(src, dst)
    R = np.logical_or(TT, src)
    B = np.logical_or(TT, dst)
    G = TT
    img = np.stack((B, G, R), axis=2).astype(np.float16)
    match_score = np.sum(TT) / np.sum(np.logical_or(src, dst))
    return match_score, img

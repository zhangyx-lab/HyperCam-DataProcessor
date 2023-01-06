# -*- coding: utf-8 -*-
# OS packages
import env
import math
from os.path import basename
# PIP packages
import cv2 as cv
import numpy as np
# User custom packages
from util.util import rdGray, pad, resize, gamma, gammaAlign
from util.refImage import load
# Kernel size constant
SAVE_PATH = env.ALIGNED_PATH
SIZE = env.KERNEL_SIZE


def loadKernelStack(id: str) -> np.ndarray:
    stack = ["{}_{}.png".format(id, _) for _ in env.COLORS]
    stack = [rdGray(env.CALIBRATED_PATH / _) for _ in stack]
    stack = [_.reshape((_.shape[0], _.shape[1], 1)) for _ in stack]
    return np.concatenate(stack, axis=2)


def getKernel(stack: np.ndarray) -> np.ndarray:
    result = np.sum(stack, axis=2) - np.max(stack, axis=2)
    result = (result / np.max(result) * 255).astype(np.uint8)
    return cv.resize(result, (SIZE, SIZE))


def apply(id):
    log = open(env.VAR_PATH / "align.txt", 'a')
    stack = loadKernelStack(id)
    kernel = getKernel(stack)
    # Find best match (highest confidence)
    best_score = 0
    best_pos = (0, 0)
    best_id = None
    for refFile in env.REF_IMAGES:
        refId = basename(refFile)
        ref = rdGray(env.VAR_PATH / "REF_{}.png".format(refId))
        # Run template matching
        match = cv.matchTemplate(ref, kernel, cv.TM_CCOEFF_NORMED)
        (H, W) = match.shape
        matchPos = np.argmax(match)
        (Y, X) = (math.floor(matchPos / W), matchPos % W)
        score = match[Y][X]
        # Check if current score is better
        if score > best_score:
            best_score = score
            best_pos = (Y, X)
            best_id = refId
    # Global matching finished
    (Y, X) = best_pos
    print("{} => ({:3d}, {:3d}) of {}, score = {:.4f}".format(
        id.ljust(3), Y, X, best_id, best_score), file=log)
    # Save matched data point
    ref_match = load(best_id)[Y:Y+SIZE, X:X+SIZE, :]
    np.save(SAVE_PATH / "{}_ref".format(id), ref_match)
    np.save(SAVE_PATH / id, stack)
    # Save a picture for manual inspection
    h, w, d = stack.shape
    kernel_rgb = np.concatenate(
        [stack[:, :, _].reshape((h, w, 1)) for _ in [6, 2, 1]],
        axis=2
    )
    kernel_rgb = cv.resize(kernel_rgb, (SIZE, SIZE))
    ref_rgb = cv.imread(str(env.REF_PATH / "{}.png".format(best_id)))
    P = 0.7
    for i in range(3):
        layer = (gamma(kernel_rgb[:, :, i]).astype(np.float32) / 255) * 2 - 1
        layer[layer > 0] = layer[layer > 0] ** P
        layer[layer < 0] = -((-layer[layer < 0]) ** P)
        kernel_rgb[:, :, i] = (255 * (layer + 1) / 2).astype(np.uint8)
    kernel_rgb[:, :] = gammaAlign(kernel_rgb[:, :], ref_rgb[:, :])
    for i in range(3):
        ref_rgb[:, :, i] = gammaAlign(ref_rgb[:, :, i], kernel_rgb[:, :, i])
    ref_roi = ref_rgb[Y:Y+SIZE, X:X+SIZE, :]
    pair = (ref_roi, kernel_rgb)
    H = max(pair[0].shape[0], pair[1].shape[0])
    W = max(pair[0].shape[1], pair[1].shape[1])
    COLOR2 = (192, 64, 32)
    COLOR = (64, 32, 192)
    # pair = resize(pair[0], h=H), resize(pair[1], h=H)
    pair = (
        pad(pair[0], h=H+6, w=W+6, color=COLOR),
        pad(pair[1], h=H+6, w=W+6, color=COLOR2)
    )
    H = max(pair[0].shape[0], pair[1].shape[0])
    W = max(pair[0].shape[1], pair[1].shape[1])
    MARGIN = 60
    pair = (
        pair[0],
        np.zeros((H, MARGIN, 3)),
        pair[1]
    )
    pair_rgb = np.concatenate(pair, axis=1)
    H, W, D = pair_rgb.shape
    pair_rgb = pad(pair_rgb, h=H+2*MARGIN, w=W+2*MARGIN, color=(0, 0, 0))
    roi_rgb = cv.rectangle(np.copy(ref_rgb), (X, Y),
                           (X+SIZE, Y+SIZE), COLOR, 3)
    roi_rgb = resize(roi_rgb, w=pair_rgb.shape[1]-2*MARGIN)
    H, W, D = roi_rgb.shape
    roi_rgb = pad(roi_rgb, h=H+2*MARGIN, w=W+2 *
                  MARGIN, color=(0, 0, 0))[MARGIN:]
    img = np.concatenate((pair_rgb, roi_rgb), axis=0)
    # print(pair_rgb.shape, roi_rgb.shape, img.shape, img.dtype)
    cv.imwrite(str(SAVE_PATH / "{}.png".format(id)), img)

# -*- coding: utf-8 -*-
# OS packages
import env
import math
from os.path import basename
# PIP packages
import cv2 as cv
import numpy as np
# User custom packages
from util.util import loadStack, rdGray, pad, resize, gamma, gammaAlign
from util.refImage import load
from util.score import score as getScore
# Kernel size constant
SAVE_PATH = env.ALIGNED_PATH
SIZE = env.KERNEL_SIZE


def getKernel(stack: np.ndarray) -> np.ndarray:
    result = np.sum(stack, axis=2) - np.max(stack, axis=2)
    result = (result / np.max(result) * 255).astype(np.uint8)
    return cv.resize(result, (SIZE, SIZE))


def apply(id):
    log = open(env.REPORT_PATH, 'a')
    stack = loadStack(id)
    kernel = getKernel(stack)
    kernel = gamma(cv.equalizeHist(kernel), 2)
    # Find best match (highest confidence)
    best_score = 0
    best_pos = (0, 0)
    best_id = None
    for refFile in env.REF_IMAGES():
        refId = basename(refFile)
        ref = rdGray(env.VAR_PATH / "REF_{}.png".format(refId))
        ref = gamma(cv.equalizeHist(ref), 2)
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
    ref_match = load(best_id)[Y:Y+SIZE, X:X+SIZE, :]
    # Report score of the best match
    match_score, match_diff = getScore(ref_match[:, :, 100:160], kernel)
    print("{} => ({:3d}, {:3d}) of {}, score = {:.4f} ({:.4f})".format(
        id.ljust(3), Y, X, best_id, match_score, best_score), file=log)
    # cv.imwrite(str(SAVE_PATH / "{}_diff.png".format(id)), match_diff)
    # Save matched data point
    np.save(SAVE_PATH / "{}_REF".format(id), ref_match)
    np.save(SAVE_PATH / id, stack)
    # Save a picture for manual inspection
    h, w, d = stack.shape
    kernel_rgb = np.stack(
        # [stack[:, :, _] for _ in [6, 2, 1]],
        [kernel for _ in range(3)],
        axis=2
    )
    kernel_rgb = cv.resize(kernel_rgb, (SIZE, SIZE))
    ref_rgb = cv.imread(str(env.REF_PATH / "{}.png".format(best_id)))
    # P = 0.7
    # for i in range(3):
    #     layer = (gamma(kernel_rgb[:, :, i]).astype(np.float32) / 255) * 2 - 1
    #     layer[layer > 0] = layer[layer > 0] ** P
    #     layer[layer < 0] = -((-layer[layer < 0]) ** P)
    #     kernel_rgb[:, :, i] = (255 * (layer + 1) / 2).astype(np.uint8)
    # kernel_rgb[:, :] = gammaAlign(kernel_rgb[:, :], ref_rgb[:, :])
    # for i in range(3):
    #     ref_rgb[:, :, i] = gammaAlign(ref_rgb[:, :, i], kernel_rgb[:, :, i])
    ref_roi = np.average(ref_rgb[Y:Y+SIZE, X:X+SIZE, :], axis=2).astype(np.uint8)
    ref_roi = gamma(cv.equalizeHist(ref_roi), 2)
    ref_roi = np.stack([ref_roi for _ in range(3)], axis=2)
    row = [ref_roi, kernel_rgb, match_diff]
    LINE_WIDTH = 6
    H = max([_.shape[0] for _ in row]) + LINE_WIDTH
    W = max([_.shape[1] for _ in row]) + LINE_WIDTH
    COLORS = [(64, 32, 192), (192, 64, 32), (164, 255, 252)]
    # pair = resize(pair[0], h=H), resize(pair[1], h=H)
    row = [pad(row[i], h=H, w=W, color=COLORS[i]) for i in range(len(row))]
    # Make the margin
    MARGIN = 60
    row = [
        row[0],
        np.zeros((H, MARGIN, 3)),
        row[1],
        np.zeros((H, MARGIN, 3)),
        row[2]
    ]
    row = np.concatenate(row, axis=1)
    H, W, D = row.shape
    row = pad(row, h=H+2*MARGIN, w=W+2*MARGIN, color=(0, 0, 0))
    roi_rgb = cv.rectangle(np.copy(ref_rgb), (X, Y),
                           (X+SIZE, Y+SIZE), COLORS[0], 3)
    roi_rgb = resize(roi_rgb, w=row.shape[1]-2*MARGIN)
    H, W, D = roi_rgb.shape
    roi_rgb = pad(roi_rgb, h=H+2*MARGIN, w=W+2 *
                  MARGIN, color=(0, 0, 0))[MARGIN:]
    img = np.concatenate((row, roi_rgb), axis=0)
    # print(row.shape, roi_rgb.shape, img.shape, img.dtype)
    cv.imwrite(str(SAVE_PATH / "{}.png".format(id)), img)

# OS packages
import env
import math
from os.path import basename
# PIP packages
import cv2 as cv
import numpy as np
from numpy.typing import NDArray as NPA
# User custom packages
from util.info import INFO, runtime_info, runtime_log
from util.util import loadStack, rdGray, pad, resize, gamma, gammaAlign
from util.score import score as getScore
from util.convert import REF2GRAY, REF2BGR, OUR2BGR
from param import DTYPE, DTYPE_MAX
# Path to save results from current module
SAVE_PATH = env.ALIGNED_PATH
env.ensureDir(SAVE_PATH / 'raw')
env.ensureDir(SAVE_PATH / 'ref')
# Get template match kernel size
getInfo = INFO("RawImage.Reference")
getRuntimeInfo = INFO("Runtime", reload=True)



def loadRef(refID: str) -> NPA[np.float32]:
    ref = np.load(env.VAR_PATH / f"REF_{refID}.npy")
    assert ref.dtype == np.float32
    return ref


def init():
    our_pd = getRuntimeInfo("pixel-density-our-camera", float)
    our_size = getRuntimeInfo("cropping-area-size", int)
    ref_pd = getInfo("pixel-density", float)
    kernel_size = int(ref_pd * our_size / our_pd)
    runtime_log(
        f"Kernel size is computed using:",
        f"  - our camera's pixel density      -> {our_pd}",
        f"  - our camera's cropped image size -> {our_size}",
        f"  - ref camera's piexl density      -> {ref_pd}",
    )
    runtime_info("kernel-size", kernel_size)


def getKernel(stack: np.ndarray) -> np.ndarray:
    SIZE = getRuntimeInfo("kernel-size", int)
    stack = stack.astype(np.float32) / DTYPE_MAX
    # Remove LED spot and dark image(s)
    result = np.average(np.sort(stack, axis=2)[:, :, 2:6], axis=2)
    return cv.resize(result, (SIZE, SIZE))


def findBestMatch(kernel: NPA[np.float32]):
    # Find best match (highest confidence)
    best_score = 0
    best_pos = (0, 0)
    best_id = None
    for refFile in env.REF_IMAGES():
        refID = basename(refFile)
        ref = loadRef(refID)
        ref = REF2GRAY(ref)
        # Run template matching
        match = cv.matchTemplate(ref, kernel, cv.TM_CCOEFF_NORMED)
        H, W = match.shape
        matchPos = np.argmax(match)
        Y, X = (math.floor(matchPos / W), matchPos % W)
        score = match[Y][X]
        # Check if current score is better
        if score > best_score:
            best_score = score
            best_pos = (Y, X)
            best_id = refID
    return best_id, best_pos, best_score


def apply(id):
    SIZE = getRuntimeInfo("kernel-size", int)
    log = open(env.REPORT_PATH, 'a')
    stack = loadStack(id)
    kernel = getKernel(stack)
    # Find best match (highest confidence)
    refID, pos, score = findBestMatch(kernel)
    # Global matching finished
    (Y, X) = pos
    ref_match = load(refID)[Y:Y+SIZE, X:X+SIZE, :]
    # Report score of the best match
    match_score, match_diff = getScore(ref_match[:, :, 100:160], kernel)
    print(
        f"{id.ljust(3)} => ({Y:3d}, {X:3d}) of {refID},",
        f"score = {match_score:.4f} ({score:.4f})",
        file=log
    )
    # Save matched data point
    np.save(SAVE_PATH / 'ref' / id, ref_match)
    np.save(SAVE_PATH / 'raw' / id, stack)
    # Save a picture for manual inspection
    h, w, d = stack.shape
    kernel_bgr = OUR2BGR(stack)
    kernel_bgr = cv.resize(kernel_bgr, (SIZE, SIZE))
    ref_cube = loadRef(refID)
    ref_bgr = REF2BGR(ref_cube)
    # Draw ROI
    ref_roi = np.average(ref_bgr[Y:Y+SIZE, X:X+SIZE, :], axis=2).astype(DTYPE)
    ref_roi = gamma(cv.equalizeHist(ref_roi), 2)
    ref_roi = np.stack([ref_roi for _ in range(3)], axis=2)
    row = [ref_roi, kernel_bgr, match_diff]
    LINE_WIDTH = 6
    H = max([_.shape[0] for _ in row]) + LINE_WIDTH
    W = max([_.shape[1] for _ in row]) + LINE_WIDTH
    COLORS = (np.array([
        [0.25, 0.13, 0.75],
        [0.75, 0.25, 0.13],
        [0.64, 1.00, 0.99],
    ]) * DTYPE_MAX).astype(DTYPE)
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
    roi_bgr = cv.rectangle(np.copy(ref_bgr), (X, Y),
                           (X+SIZE, Y+SIZE), COLORS[0], 3)
    roi_bgr = resize(roi_bgr, w=row.shape[1]-2*MARGIN)
    H, W, D = roi_bgr.shape
    roi_bgr = pad(roi_bgr, h=H+2*MARGIN, w=W+2 *
                  MARGIN, color=(0, 0, 0))[MARGIN:]
    img = np.concatenate((row, roi_bgr), axis=0)
    # print(row.shape, roi_bgr.shape, img.shape, img.dtype)
    cv.imwrite(str(SAVE_PATH / "{}.png".format(id)), img)

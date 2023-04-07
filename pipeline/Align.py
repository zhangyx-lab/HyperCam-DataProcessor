# OS packages
import env
import math
from os.path import basename
# PIP packages
import cvtb
import cv2 as cv
import numpy as np
from numpy.typing import NDArray as NPA
# User custom packages
import util.util as util
from util.info import INFO, runtime_info, runtime_log
from util.util import loadStack, rdGray
from util.score import score as getScore
from util.convert import REF2GRAY, REF2BGR, OUR2BGR, OUR2GRAY
from pipeline.RefImage import U8C1_PATH
# Path to save results from current module
SAVE_PATH = env.ALIGNED_PATH
env.ensureDir(SAVE_PATH / 'raw')
env.ensureDir(SAVE_PATH / 'ref')
# Get template match kernel size
getInfo = INFO("RawImage.Reference")
getRuntimeInfo = INFO("Runtime", reload=True)


def loadRef(refID: str) -> NPA[np.float32]:
    ref = np.load(env.REF_CAL_PATH / f"{refID}.npy")
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


def getKernel(stack: np.ndarray, level: float = 0.5, g: float = 2, ks=320):
    SIZE = getRuntimeInfo("kernel-size", int)
    # Get gamma lambda function
    gamma = cvtb.histogram.gamma(g)
    # Equalize histogram
    result = [cvtb.types.F32(stack[:, :, i]) for i in range(2, 8)]
    result = [l * level / np.average(l[l < 0.9]) for l in result]
    result = [gamma(l) for l in result]
    result = [l * level / np.average(l) for l in result]
    # Remove LED spot and dark image(s)
    result = np.stack(result, axis=2)
    result = np.min(result, axis=2)
    # Equalize geometrical intensity distribution
    if ks:
        result = cvtb.geometric.equalize(ks * 2 + 1)(result)
    # stack = np.average(np.sort(stack, axis=2)[:, :, 2:4], axis=2)
    return cvtb.types.U8(cv.resize(result, (SIZE, SIZE)))


def findBestMatch(kernel: NPA[np.uint8], name='unknown'):
    kernel = cvtb.types.U8(kernel)
    # Find best match (highest confidence)
    best_score = 0
    best_pos = (0, 0)
    best_id = None
    for refFile in env.REF_IMAGES():
        refID = basename(refFile)
        # ref = loadRef(refID)
        # ref = REF2GRAY(ref)
        ref = np.load(U8C1_PATH / f"{refID}.npy")
        ref = cvtb.types.U8(cvtb.spectral.gray()(ref))
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
    kernel = rdGray(env.ALIGN_KERNEL_PATH / f"{id}.png")
    # Find best match (highest confidence)
    refID, pos, score = findBestMatch(kernel, id)
    # Global matching finished
    (Y, X) = pos
    ref_cube = loadRef(refID)
    ref_cube_match = ref_cube[Y:Y+SIZE, X:X+SIZE, :]
    # Report score of the best match
    match_score, match_diff = getScore(
        REF2GRAY(ref_cube_match), OUR2GRAY(kernel))
    print(
        f"{id.ljust(3)} => ({Y:3d}, {X:3d}) of {refID},",
        f"score = {match_score:.4f} ({score:.4f})",
        file=log
    )
    # Load stack
    stack = np.load(env.CALIBRATED_PATH / f"{id}.npy")
    # Save matched data point
    np.save(SAVE_PATH / 'ref' / id, ref_cube_match)
    np.save(SAVE_PATH / 'raw' / id, stack)
    # Save a picture for manual inspection
    kernel_bgr = cvtb.types.U8(OUR2BGR(stack))
    kernel_bgr = cv.resize(kernel_bgr, (SIZE, SIZE))
    ref_bgr = cvtb.types.U8(REF2BGR(ref_cube))
    # Draw ROI
    ref_roi = ref_bgr[Y:Y+SIZE, X:X+SIZE, :]
    # Merge image into grid
    row = [ref_roi, kernel_bgr, match_diff]
    LINE_WIDTH = 6
    MARGIN = 60
    H = max([_.shape[0] for _ in row])
    W = max([_.shape[1] for _ in row])
    COLORS = cvtb.types.to_float([
        [0.25, 0.13, 0.75],
        [0.75, 0.25, 0.13],
        [0.64, 1.00, 0.99],
    ])
    # pair = resize(pair[0], h=H), resize(pair[1], h=H)
    resize = cvtb.transform.resize(height=H)
    row = list(map(resize, row))
    row = [
        cvtb.misc.pad(LINE_WIDTH, color=COLORS[i])(row[i])
        for i in range(len(row))
    ]
    # Make the margin
    pad = cvtb.misc.pad(0, MARGIN)
    row[1] = pad(row[1])
    row = np.concatenate(row, axis=1)
    H, W, D = row.shape
    row = cvtb.misc.pad(MARGIN)(row)
    cv.rectangle(
        ref_bgr, (X, Y), (X+SIZE, Y+SIZE),
        list(map(int, cvtb.types.U8(COLORS[0]))), 5
    )
    ref_bgr = cvtb.transform.resize(width=W)(ref_bgr)
    ref_bgr = cvtb.misc.pad(MARGIN, top=0)(ref_bgr)
    img = np.concatenate((row, ref_bgr), axis=0)
    cv.imwrite(str(SAVE_PATH / f"{id}.png"), cvtb.types.U8(img))


if __name__ == '__main__':
    idList = util.getIdList(env.CALIBRATED_IMAGES())
    # for id in idList:
    #     print(f"Matching {id}")
    #     apply(id)
    i = 0
    stack = loadStack(idList[i])
    names = ["level", "gamma", "kernel"]
    params = [500,     100,        0]
    limits = [5000,     500,      999]
    scales = [1000,     100,        0]

    def show():
        args = [a / b if b else a for a, b in zip(params, scales)]
        kernel = getKernel(stack, *args)
        cv.imshow("kernel", cvtb.types.U8(kernel))

    def updater(index):
        def update(val):
            params[index] = val
            show()
        return update

    cv.namedWindow("kernel")
    for i, n, p, l in zip(range(len(names)), names, params, limits):
        cv.createTrackbar(n, "kernel", p, l, updater(i))
    cv.startWindowThread()

    while True:
        show()
        k = cv.waitKey(0)
        if k == ord('['):
            if i > 0:
                i -= 1
                stack = loadStack(idList[i])
        elif k == ord(']'):
            if i + 1 < len(idList):
                i += 1
                stack = loadStack(idList[i])
        else:
            break

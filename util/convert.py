import numpy as np
from numpy.typing import NDArray as NPA
from param import DTYPE, DTYPE_MAX, REF_BANDS, LED_LIST
from util.util import gamma, contrast


def gaussian(x, u, sigma): return np.exp(-np.square((u - x) / (sigma ** 2)))


def trimToFit(cube: NPA[np.float32], lim=[0, 1]) -> NPA[np.float32]:
    # Trim left side
    cube[cube < lim[0]] = lim[0]
    # Trim right side
    cube[cube > lim[1]] = lim[1]
    return cube


def scaleToFit(cube: NPA[np.float32], lim=[0, 1]) -> NPA[np.float32]:
    # Align left size to 0
    cube -= np.min(cube)
    # Scale to fit range
    cube *= lim[1] - lim[0]
    # Shift to align with lim
    cube += lim[0]
    return cube


def checkOverFlow(cube: NPA[np.float32], lim=[0, 1], handler=None):
    if handler is None:
        assert np.min(cube) >= lim[0], np.min(cube)
        assert np.max(cube) <= lim[1], np.max(cube)
        return cube
    else:
        return handler(cube, lim)


def REF2BAND(cube: NPA[np.float32], band: float, sigma: float = 1) -> NPA[np.float32]:
    assert sigma != 0
    x = np.array(REF_BANDS)
    return np.average(cube, axis=2, weights=gaussian(x, band, sigma))


def REF2BGR(cube: NPA[np.float32], overflow=trimToFit) -> NPA[np.uint8]:
    cube = checkOverFlow(cube, handler=overflow)
    # Match the band number with the LED module
    bgr = np.stack([
        gamma(REF2BAND(cube, LED_LIST[i].bandwidth, LED_LIST[i].delta / 10), g)
        for i, g in zip([1, 2, 6], [1.15, 1.18, 1.55])
    ], axis=2)
    bgr *= 0.3 / np.average(bgr)
    bgr = np.rint(trimToFit(bgr) * 255).astype(np.uint8)
    return bgr


def REF2GRAY(cube: NPA[np.float32], overflow=trimToFit) -> NPA[np.float32]:
    cube = checkOverFlow(cube, handler=overflow)
    x = np.array(REF_BANDS)
    gray = np.average(cube, axis=2, weights=gaussian(x, 500, 10))
    return gray


def OUR2BGR(cube: NPA[DTYPE]) -> NPA[np.uint8]:
    bgr = cube[:, :, [1, 2, 6]]
    bgr = (bgr >> 8).astype(np.uint8)
    return bgr


def OUR2GRAY(cube: NPA[DTYPE]) -> NPA[np.uint8]:
    gray = np.average(cube, axis=2)
    gray = np.rint(255 * gray / DTYPE_MAX).astype(np.uint8)
    return gray

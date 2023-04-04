import numpy as np
from numpy.typing import NDArray as NPA
from param import DTYPE, DTYPE_MAX, REF_BANDS, LED_LIST
import util.util as util


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


def U8(img: NPA, overflow=trimToFit) -> NPA[np.uint8]:
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    match img.dtype:
        case np.uint8: return img
        case np.uint16: return (img >> 8).astype(np.uint8)
        case t:
            assert issubclass(t.type, np.floating), t
            img = np.rint(checkOverFlow(img, handler=overflow) * 255)
            return img.astype(np.uint8)


def U16(img: NPA, overflow=trimToFit) -> NPA[np.uint16]:
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    match img.dtype:
        case np.uint8: return (img << 8).astype(np.uint16)
        case np.uint16: img
        case t:
            assert issubclass(t.type, np.floating), t
            img = np.rint(checkOverFlow(img, handler=overflow) * 65535)
            return img.astype(np.uint16)


def F32(img: NPA) -> NPA[np.float32]:
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    match img.dtype:
        case np.uint8: return img.astype(np.float32) / 255
        case np.uint16: return img.astype(np.float32) / 65535
    return img.astype(np.float32)



def REF2BAND(cube: NPA[np.float32], band: float, sigma: float = 1):
    assert sigma != 0
    x = np.array(REF_BANDS)
    band = np.average(cube, axis=2, weights=gaussian(x, band, sigma))
    return F32(band)


def REF2BGR(cube: NPA[np.float32], overflow=trimToFit):
    cube = checkOverFlow(cube, handler=overflow)
    # Match the band number with the LED module
    bgr = np.stack([
        util.gamma(REF2BAND(cube, LED_LIST[i].bandwidth, LED_LIST[i].delta / 10), g)
        for i, g in zip([1, 2, 6], [1.15, 1.18, 1.55])
    ], axis=2)
    bgr *= 0.3 / np.average(bgr)
    return U8(bgr, trimToFit)


def REF2GRAY(cube: NPA, overflow=trimToFit):
    if len(cube.shape) < 3:
        return cube
    x = np.array(REF_BANDS)
    gray = np.average(cube, axis=2, weights=gaussian(x, 500, 10))
    return checkOverFlow(F32(gray), handler=overflow)


def OUR2BGR(cube: NPA):
    # bgr = cube[:, :, [1, 2, 6]]
    bgr = np.stack([
        util.gamma(F32(cube[:, :, i]) * 2, g)
        for i, g in zip([1, 2, 6], [1.25, 1.28, 1.58])
    ], axis=2)
    # bgr *= 0.3 / np.average(bgr)
    return U8(bgr)


def OUR2GRAY(cube: NPA):
    if len(cube.shape) < 3:
        return cube
    gray = np.average(cube, axis=2)
    return F32(gray)

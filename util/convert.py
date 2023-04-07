import cvtb
import numpy as np
from numpy.typing import NDArray
from param import REF_BANDS, LED_LIST
import util.util as util


def post_process_bgr(bgr: NDArray):
    return cvtb.types.scaleToFitDR(bgr)
    # FROM REF
    # Match the band number with the LED module
    bgr = np.stack([
        gamma(
            REF2BAND(cube, LED_LIST[i].bandwidth, LED_LIST[i].delta / 10), g)
        for i, g in zip([1, 2, 6], [1.15, 1.18, 1.55])
    ], axis=2)
    bgr *= 0.3 / np.average(bgr)
    # FROM OUR
    bgr = np.stack([
        gamma(F32(cube[:, :, i]) * 2, g)
        for i, g in zip([1, 2, 6], [1.25, 1.28, 1.58])
    ], axis=2)


def REF2BAND(cube: NDArray[np.float32], band: float, sigma: float = 1):
    gaussian = cvtb.fx.gaussian(band, sigma)
    gray = cvtb.spectral.gray(weights=gaussian, bands=np.array(REF_BANDS))
    return gray(cube)


def REF2BGR(cube: NDArray[np.float32]):
    bgr = np.stack([
        REF2BAND(cube, LED_LIST[i].bandwidth, LED_LIST[i].delta / 10)
        for i in [1, 2, 6]
    ], axis=2)
    return post_process_bgr(bgr)


REF2GRAY = cvtb.spectral.gray(
    weights=cvtb.fx.gaussian(500, 10),
    bands=np.array(REF_BANDS)
)


def OUR2BAND(cube: NDArray[np.float32], band: float, sigma: float = 1):
    gaussian = cvtb.fx.gaussian(band, sigma)
    bands = np.array([led.bandwidth for led in LED_LIST])
    gray = cvtb.spectral.gray(weights=gaussian, bands=bands)
    return gray(cube)


def OUR2BGR(cube: NDArray):
    bgr = np.stack([
        OUR2BAND(cube, LED_LIST[i].bandwidth, LED_LIST[i].delta / 10)
        for i in [1, 2, 6]
    ], axis=2)
    return post_process_bgr(bgr)


OUR2GRAY = cvtb.spectral.gray()

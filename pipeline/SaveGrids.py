# PIP packages
import cvtb
import cv2 as cv
import numpy as np
# Project Packages
import env
from param import LED_LIST
from util.util import getIdList, loadStack
from util.convert import OUR2BGR
RGB_SAVE_PATH = env.ensureDir(env.GRID_VIEW_PATH / "RGB")
GRID_SAVE_PATH = env.ensureDir(env.GRID_VIEW_PATH / "GRIDS")


def apply(stack, path):
    stack = cvtb.types.U8(stack).copy()
    # Generate color RGB image
    cv.imwrite(
        str(RGB_SAVE_PATH / path),
        cvtb.types.U8(OUR2BGR(stack))
    )
    # Generate fake color grids
    MARGIN = 128
    H, W, D = stack.shape
    # Convert stack into list of grayscale layers
    stack = [stack[:, :, i] for i in range(D)]
    for i in range(len(stack)):
        name, wavelength, delta = LED_LIST[i]
        title = "{} ({} ~ {} nm)".format(name, wavelength, delta)
        color = cvtb.types.F32(cvtb.spectral.wave2bgr(wavelength))
        # Generate layer slices
        layer = cvtb.types.F32(np.stack([stack[i]] * 3, axis=2))
        layer = layer * 0.5 / np.average(layer)
        layer = cvtb.histogram.gamma(2)(layer)
        layer = layer * 0.5 / np.average(layer)
        layer = layer * color
        layer = cvtb.misc.pad(MARGIN, top=0)(layer)
        # Put suffix text
        t_color = np.minimum(color, 100 * color / np.average(color))
        B, G, R = cvtb.types.U8(t_color)
        text = cvtb.misc.text(
            color=(B, G, R),
            scale=2, width=3,
            pos=(int(MARGIN/2), layer.shape[0] - int(MARGIN/2))
        )
        stack[i] = text(cvtb.types.U8(layer), title)
    # Concatenate all grids
    img = np.concatenate([
        np.concatenate(stack[:4], axis=1),
        np.concatenate(stack[4:], axis=1)
    ], axis=0)
    # Pad Outer Frame
    H, W, D = img.shape
    img = cvtb.misc.pad(top=MARGIN)(img)
    # Save
    cv.imwrite(
        str(GRID_SAVE_PATH / path),
        cvtb.types.U8(img, cvtb.types.scaleToFit)
    )


if __name__ == '__main__':
    for id in getIdList(env.CALIBRATED_IMAGES()):
        stack = loadStack(id, env.RAW_PATH)
        apply(stack, f"{id}_Raw.png")
        stack = loadStack(id, env.CALIBRATED_PATH)
        apply(stack, f"{id}_Cal.png")

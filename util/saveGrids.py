# PIP packages
import cv2 as cv
import numpy as np
# Project Packages
import env
from param import LED_LIST
from util.util import pad, draw_text, wave2bgr, gamma, getIdList, loadStack
from util.convert import F32, U8, OUR2BGR, scaleToFit
RGB_SAVE_PATH = env.ensureDir(env.GRID_VIEW_PATH / "RGB")
GRID_SAVE_PATH = env.ensureDir(env.GRID_VIEW_PATH / "GRIDS")


def apply(stack, path):
    stack = U8(stack).copy()
    # Generate color RGB image
    cv.imwrite(
        str(RGB_SAVE_PATH / path),
        U8(OUR2BGR(stack))
    )
    # Generate fake color grids
    MARGIN = 128
    H, W, D = stack.shape
    # Convert stack into list of grayscale layers
    stack = [stack[:, :, i] for i in range(D)]
    for i in range(len(stack)):
        name, wavelength, delta = LED_LIST[i]
        title = "{} ({} ~ {} nm)".format(name, wavelength, delta)
        color = F32(wave2bgr(wavelength))
        # Generate layer slices
        layer = F32(np.stack([stack[i]] * 3, axis=2))
        layer = layer * 0.5 / np.average(layer)
        layer = gamma(layer, 2)
        layer = layer * 0.5 / np.average(layer)
        layer = layer * color
        layer = pad(U8(layer), h=H+2*MARGIN, w=W+MARGIN)[MARGIN:]
        # Put suffix text
        t_color = np.minimum(color, 100 * color / np.average(color))
        B, G, R = U8(t_color)
        stack[i] = draw_text(
            layer, title, color=(B, G, R),
            scale=2, width=3,
            pos=(int(MARGIN/2), layer.shape[0] - int(MARGIN/2))
        )
    # Concatenate all grids
    img = np.concatenate([
        np.concatenate(stack[:4], axis=1),
        np.concatenate(stack[4:], axis=1)
    ], axis=0)
    # Pad Outer Frame
    H, W, D = img.shape
    img = pad(img, h=H+2*MARGIN, w=W+MARGIN)[:-MARGIN, MARGIN:-MARGIN]
    # Save
    cv.imwrite(
        str(GRID_SAVE_PATH / path),
        U8(img, scaleToFit)
    )


if __name__ == '__main__':
    for id in getIdList(env.CALIBRATED_IMAGES()):
        stack = loadStack(id, env.RAW_PATH)
        apply(stack, f"{id}_Raw.png")
        stack = loadStack(id, env.CALIBRATED_PATH)
        apply(stack, f"{id}_Cal.png")

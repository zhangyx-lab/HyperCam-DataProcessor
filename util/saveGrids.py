# PIP packages
import cv2 as cv
import numpy as np
from param import LED_LIST
from util.util import pad, draw_text, wave2bgr

def apply(stack, path):
    MARGIN = 128
    H, W, D = stack.shape
    stack = [stack[:, :, i] for i in range(D)]
    for i in range(len(stack)):
        name, wavelength, delta = LED_LIST[i]
        title = "{} ({} ~ {} nm)".format(name, wavelength, delta)
        color = wave2bgr(wavelength)
        layer = np.stack([stack[i] for _ in range(3)],
                         axis=2).astype(np.float64)
        layer = (layer / np.max(layer)) * color
        layer = pad(layer, h=H+2*MARGIN, w=W+MARGIN).astype(np.uint8)[MARGIN:]
        t_color = color.astype(np.float64)
        t_color = np.minimum(t_color, 100 * t_color / np.average(t_color))
        B, G, R = [int(_) for _ in t_color]
        stack[i] = draw_text(
            layer, title, color=(B, G, R),
            scale=2, width=3,
            pos=(int(MARGIN/2), layer.shape[0] - int(MARGIN/2))
        )
    img = np.concatenate([
        np.concatenate(stack[:4], axis=1),
        np.concatenate(stack[4:], axis=1)
    ], axis=0)
    H, W, D = img.shape
    img = pad(img, h=H+2*MARGIN, w=W+MARGIN)[:-MARGIN, MARGIN:-MARGIN]
    cv.imwrite(str(path), img)

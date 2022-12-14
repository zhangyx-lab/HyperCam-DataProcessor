# -*- coding: utf-8 -*-
import re
import cv2
import sys
import glob
import math
import numpy as np
import spectral.io.envi as envi


path = "./coffee_ref/coffee_ref"
data = envi.open(path + ".hdr", path + ".dat")
stack = np.array(data.load())


# WINDOW_NAME = "Reference Image"
# cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
# cv2.startWindowThread()

(H, W, D) = stack.shape
for i in range(D):
    stack[:, :, i] = stack[:, :, i] / np.max(stack[:, :, i])
    # cv2.imshow(WINDOW_NAME, (stack[:, :, i] * 255).astype(np.uint8))
    # cv2.waitKey(33)

# cv2.waitKey(1000)
# cv2.destroyWindow(WINDOW_NAME)
# cv2.waitKey(10)

ref = (np.squeeze(np.average(stack, axis=2)) * 255).astype(np.uint8)
# cv2.imwrite("normalized_coffee_ref.png", ref)
ref_rgb = cv2.imread(path + ".png")

samples = list(glob.glob("./results/*"))
samples.sort()

keys = list(map(lambda s: re.findall("(?<=\/)[A-Z]\d+(?=_)", s)[0], samples))
samples = list(map(lambda p: cv2.imread(p), samples))

lut = dict.fromkeys(keys)

for key in lut:
    lut[key] = []

for i in range(len(keys)):
    key = keys[i]
    img = samples[i]
    lut[key].append(img)


WINDOW_NAME = "Sample Image"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
cv2.startWindowThread()


WINDOW_NAME2 = "Comparison"
cv2.namedWindow(WINDOW_NAME2, cv2.WINDOW_AUTOSIZE)
cv2.startWindowThread()


WINDOW_NAME3 = "LOC MAP"
cv2.namedWindow(WINDOW_NAME3, cv2.WINDOW_AUTOSIZE)
cv2.startWindowThread()

SIZE = 286

for key in lut:
    print(key)
    stack = lut[key]
    stack = [img.astype(np.float64) for img in stack]
    stack = [img / np.max(img) for img in stack]
    stack = np.concatenate(stack, axis=2)
    result = np.sum(stack, axis=2) - np.max(stack, axis=2)
    result = (result / np.max(result) * 255).astype(np.uint8)
    # Downscale samples from 1024px to 286px
    kernel = cv2.resize(result, (SIZE, SIZE))
    loc_map = cv2.matchTemplate(ref, kernel, cv2.TM_CCOEFF_NORMED)
    (H, W) = loc_map.shape
    pos = np.argmax(loc_map)
    (Y, X) = (math.floor(pos / W), pos % W)
    # ref_rgb = np.reshape(np.stack((ref) * 3, axis=-1), (H, W, 3))
    loc_rgb = cv2.rectangle(np.copy(ref_rgb), (X, Y), (X + SIZE, Y + SIZE), (255, 255, 0), 2)
    cv2.imshow(WINDOW_NAME, loc_rgb)
    ref_match = ref[Y:Y+SIZE, X:X+SIZE]
    ref_match = 255 * ref_match.astype(np.float32) / np.max(ref_match)
    cv2.imshow(
        WINDOW_NAME2,
        np.concatenate(
            [
                ref_match.astype(np.uint8),
                np.zeros((SIZE, 10), dtype=np.uint8),
                kernel
            ],
            axis=1
        )
    )
    cv2.imshow(WINDOW_NAME3, loc_map / np.max(loc_map))
    key = cv2.waitKey(0)
    if key == ord('q') or key == ord('Q') or key == 9:
        break

cv2.destroyWindow(WINDOW_NAME)
cv2.destroyWindow(WINDOW_NAME2)
cv2.destroyWindow(WINDOW_NAME3)
cv2.waitKey(10)

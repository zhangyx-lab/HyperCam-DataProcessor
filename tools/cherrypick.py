from glob import glob
from os.path import basename, exists
# PIP Packages
import cv2
import numpy as np
# Project Includes
import env
import util.util as util
# Initialize lists
LIST_FILE = env.ALIGNED_PATH / 'list.txt'
l = glob(str(env.ALIGNED_PATH / "*.png"))
LIST = [s.replace('.png', '') for s in map(basename, l)]
LIST.sort()
if exists(LIST_FILE):
    lines = open(LIST_FILE, 'r').readlines()
    LIST_PREV = [l.replace("\n", "") for l in lines]
else:
    LIST_PREV = []
# Initialize table of states
SELECT = np.array(
    [id in LIST_PREV for id in LIST],
    dtype=np.bool8
)
# State management
cursor = 0
image = None
# Renderer


def loadImage(cursor):
    global image

    def exclude(arr, val=0):
        i = 0
        j = len(arr)
        while arr[i] == val:
            i += 1
        while arr[j - 1] == val:
            j -= 1
        assert i < j
        return i, j

    def trimImage(img):
        gary = np.max(img, axis=2)
        y1, y2 = exclude(np.max(gary, axis=1))
        x1, x2 = exclude(np.max(gary, axis=0))
        return img[y1:y2, x1:x2]

    if image is None:
        id = LIST[cursor]
        path = env.ALIGNED_PATH / f"{id}.png"
        im = cv2.imread(str(path))
        image = (
            trimImage(im[:400]),
            trimImage(im[400:]),
        )
    return image


WIN = "Cherry Pick"
WINR = "Cherry Pick - Reference"
WINS = "Cherry Pick - Selection"
cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
cv2.namedWindow(WINR, cv2.WINDOW_AUTOSIZE)
cv2.namedWindow(WINS, cv2.WINDOW_AUTOSIZE)


def selection(c):
    text = LIST[c]
    if SELECT[c]:
        text += ' Selected'
    selection = [
        np.zeros(SELECT.shape, dtype=np.uint8)
        for _ in range(3)
    ]
    selection[1][SELECT] = 255
    selection[2][~SELECT] = 255
    selection[1][c] &= 127
    selection[2][c] &= 127
    img = np.stack(selection, axis=-1)
    img = np.stack([img] * 10, axis=0)
    h, w, _ = img.shape
    W = w * 10
    img = cv2.resize(img, (W, h), interpolation=cv2.INTER_NEAREST)
    im2 = ~np.zeros((60, W, _), img.dtype)
    im2 = util.draw_text(im2, text, color=[0] * 3, scale=2, width=2)
    return np.concatenate([im2, img], axis=0)


def render():
    global cursor
    cv2.imshow(WINS, selection(cursor))
    img1, img2 = loadImage(cursor)
    cv2.imshow(WIN, img1)
    cv2.imshow(WINR, img2)


def update_cursor(val):
    global cursor, image
    cursor = val
    image = None
    render()


def toggle():
    global cursor
    SELECT[cursor] = not SELECT[cursor]
    render()


# Start OpenCV window session
cv2.createTrackbar(
    "navigate", WIN,
    cursor, len(LIST),
    update_cursor
)
cv2.startWindowThread()
# Key stroke handler
render()
while True:
    key = cv2.waitKey(0)
    if key == ord('q'):
        exit(0)
    elif key == ord(' '):
        # toggle selection
        toggle()
    elif key == ord('[') and cursor > 0:
        cv2.setTrackbarPos(
            "navigate", WIN,
            cursor - 1
        )
    elif key == ord(']') and cursor + 1 < len(LIST):
        cv2.setTrackbarPos(
            "navigate", WIN,
            cursor + 1
        )
    else:
        if key == ord('s') or key == 13: # ENTER = 13
            break
        else:
            print(key)
# Save selection
with open(LIST_FILE, 'w') as f:
    for ID, select in zip(LIST, list(SELECT)):
        if (select):
            f.write(ID + '\n')

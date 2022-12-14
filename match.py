# Python program to illustrate
# template matching
# SCALE => 286
import cv2
import numpy as np

# Read the main image
img_rgb = cv2.imread('ref.png')

# Convert it to grayscale
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# Read the template
template = cv2.cvtColor(cv2.imread('results/A1_Blue.png'), cv2.COLOR_BGR2GRAY)

# Store width and height of template in w and h
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

# Specify a threshold
threshold = 0.1

# Store the coordinates of matched area in a numpy array
loc = np.where(res >= threshold)

# Draw a rectangle around the matched region.
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)

# Show the final image with the matched area.
cv2.namedWindow('Detected', cv2.WINDOW_AUTOSIZE)
cv2.startWindowThread()
cv2.imshow('Detected', img_rgb)
cv2.waitKey(0)
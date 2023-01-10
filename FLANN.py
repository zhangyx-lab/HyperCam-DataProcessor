import cv2 as cv
import numpy as np
import env
from util.util import gamma, contrast, resize
img1_path = env.VAR_PATH / "REF_L1.png"
img2_path = env.CALIBRATED_PATH / "B6_Red.png"
print(img1_path, img2_path)
img1 = cv.imread(str(img1_path), cv.IMREAD_GRAYSCALE)
img2 = cv.imread(str(img2_path), cv.IMREAD_GRAYSCALE)
# Correction
img1 = gamma(cv.equalizeHist(img1), 2)
img2 = resize(img2, env.KERNEL_SIZE)
img2 = gamma(cv.equalizeHist(img2), 2)
# img2 = contrast(img2, 2)
# -- Step 1: Detect the keypoints using SIFT Detector, compute the descriptors
detector = cv.SIFT_create()
keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
keypoints2, descriptors2 = detector.detectAndCompute(img2, None)
# -- Step 2: Matching descriptor vectors with a FLANN based matcher
# Since SURF is a floating-point descriptor NORM_L2 is used
matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)
# -- Filter matches using the Lowe's ratio test
ratio_thresh = 0.5
good_matches = []
for m, n in knn_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)
# -- Draw matches
img_matches = np.empty(
    (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches,
               img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# -- Show detected matches
cv.imshow('Good Matches', img_matches)
cv.waitKey()

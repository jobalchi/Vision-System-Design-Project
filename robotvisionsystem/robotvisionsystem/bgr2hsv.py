import cv2 as cv
import numpy as np

bgr_range = np.uint8([[[128, 128, 128]]])
cvt_hsv_range = cv.cvtColor(bgr_range, cv.COLOR_BGR2HSV)
print(cvt_hsv_range)

cvt_bgr_range = cv.cvtColor(cvt_hsv_range, cv.COLOR_HSV2BGR)
print(cvt_bgr_range)

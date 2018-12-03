import cv2
import numpy as np

def nothingLeftOnImg(img, imgSize):
    zeros = imgSize - cv2.countNonZero(img)
    return zeros == imgSize

def skeletonize(imgThresholded):
    imgSize = np.size(imgThresholded)
    skel = np.zeros(imgThresholded.shape, np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

    while(True):
        eroded = cv2.erode(imgThresholded, kernel)
        dilated = cv2.dilate(eroded, kernel)
        contour = cv2.subtract(imgThresholded, dilated)
        skel = cv2.bitwise_or(skel, contour)

        if nothingLeftOnImg(imgThresholded, imgSize):
            return skel
        else:
            imgThresholded = eroded.copy()

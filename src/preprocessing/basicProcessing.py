import cv2
import numpy as np

minBright = 0
maxBright = 255

def convertToGrayscale(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgGray = cv2.normalize(imgGray, None, minBright, maxBright, norm_type=cv2.NORM_MINMAX)
    return imgGray

def threshold(filePath):
    img = cv2.imread(filePath)
    imgGray = convertToGrayscale(img)

    threshold = 140
    _, imgThresholded = cv2.threshold(imgGray, threshold, maxBright, cv2.THRESH_BINARY_INV)
    return imgThresholded


def thresholdAdaptive(imgGray):
    maskSize = 25
    cParam = 2
    imgThresholded = cv2.adaptiveThreshold(imgGray, maxBright, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY_INV, maskSize, cParam)
    return imgThresholded

#filters out small blobs, connects neighbouring blobs
def openAndClose(img):
    kernel = np.ones((2, 2),np.uint8)
    #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img
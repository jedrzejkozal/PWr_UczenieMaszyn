import cv2
from preprocessing.basicProcessing import thresholdAdaptive, openAndClose

def extractThresholdedPalm(imgGray):
    imgThresholded = thresholdAdaptive(imgGray)
    imgThresholded = openAndClose(imgThresholded)
    imgPalm = extractBiggestBlob(imgThresholded)
    return cv2.resize(imgPalm, ( 25, 25 ), interpolation = cv2.INTER_NEAREST )


def extractBiggestBlob(img):
    _ ,contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        biggestContour = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(biggestContour)
        return img[y : y + h, x : x + w]
    return img

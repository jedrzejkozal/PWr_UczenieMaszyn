import numpy as np
from random import shuffle

from preprocessing import *

from utils import saveImg, convertListOfNdarraysToNdarray
import cv2
import os


class Db:
    def __init__(self, dataBaseName, saveImages = False):
        self.__saveImages = saveImages
        self.__numClasses = 10
        self.__outputImgSideLen = 100
        self.__dataBaseName = dataBaseName

    def getGrayscale(self):
        noOp = lambda img : img
        return self.__transform(noOp)

    def getThresholed(self):
        return self.__transform(contourSearching.extractThresholdedPalm)

    def getSkeletonized(self):
        extractSkeleton = lambda img : skeletonizeImg.skeletonize(contourSearching.extractThresholdedPalm(img))
        return self.__transform(extractSkeleton)

    def __transform(self, processOneImg):
        x = []
        y = []

        for classId in range(self.__numClasses):
            directory = "../db/" + self.__dataBaseName + "/" + str(classId)

            filesInDir = os.listdir(directory)
            shuffle(filesInDir)
            for file in filesInDir:
                filePath = os.path.join(directory, file)

                img = cv2.imread(filePath)
                imgGray = basicProcessing.convertToGrayscale(img)

                if imgGray.shape != (100, 100): #IMG_5935.JPG, IMG_5874.JPG, IMG_5978.JPG
                    imgGray = cv2.resize(imgGray, (100, 100), interpolation = cv2.INTER_NEAREST )

                imgProcessed = processOneImg(imgGray)

                outputImgSize = (self.__outputImgSideLen, self.__outputImgSideLen)
                imgProcessed = cv2.resize(imgProcessed, outputImgSize, interpolation = cv2.INTER_NEAREST )

                if self.__saveImages:
                    saveImg("log/processedImages/" + str(classId), file, imgProcessed)

                x.append(imgProcessed)
                y.append(classId)
        x = convertListOfNdarraysToNdarray(x)
        return x, y

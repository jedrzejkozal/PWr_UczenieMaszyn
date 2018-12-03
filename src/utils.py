import numpy as np
import cv2
import os

def flatten(x):
    shape = x.shape
    return np.array(x).reshape((shape[0], shape[1]*shape[2]))

def loadXY(directory):
    x = np.load(os.path.join(directory, "x.npy"))
    y = np.load(os.path.join(directory, "y.npy"))
    return x,y

def saveNpy(x, y, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    x = convertListOfNdarraysToNdarray(x)

    np.save(os.path.join(directory, "x"), x)
    np.save(os.path.join(directory, "y"), y)


def saveImg(outputDir, filename, img):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    cv2.imwrite(os.path.join(outputDir, filename), img)

def convertListOfNdarraysToNdarray(toConvert):
    shape = [len(toConvert)]
    shape.extend(list(toConvert[0].shape))
    toConvert = np.concatenate(toConvert).reshape(shape)
    return toConvert
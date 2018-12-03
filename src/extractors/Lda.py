#from sklearn.lda import LDA
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis as LDA
import numpy as np
from utils import flatten

class Lda:
    def __init__(self, xTrain, yTrain):
        self.__engine = LDA(n_components=100)

        xFlat = self.flatten(xTrain)
        yFlat = np.array(yTrain, dtype="float64")
        print("before fit")
        self.__engine.fit(xFlat, yFlat)
        print("after fit")


    def transform(self, x):
        xFlat = flatten(x)
        return self.__engine.transform(xFlat)

    def flatten(self, x):
        shape = x.shape
        return np.array(x).reshape((shape[0], shape[1]*shape[2])).astype("float64")

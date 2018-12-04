#https://www.kaggle.com/hamishdickson/preprocessing-images-with-dimensionality-reduction

from sklearn.decomposition import PCA
import numpy as np
from utils import flatten

class Pca:
    def __init__(self, xTrain, yTrain, numberOfClasses):
        pass


    def fit(self, xTrain, yTrain):
        """
        varianceThreshold = 0.95
        self.__engine = PCA()
        xFlat = flatten(xTrain)
        """

        """
        self.__engine.fit(xFlat)
        cumsum = np.cumsum(self.__engine.explained_variance_ratio_)
        numFeatures = np.argmax(cumsum >= varianceThreshold) + 1

        self.__engine = PCA(n_components = numFeatures)
        """

        xFlat = flatten(xTrain)
        self.__engine = PCA(n_components=100)
        self.__engine.fit(xFlat)


    def transform(self, x):
        xFlat = flatten(x)
        return self.__engine.transform(xFlat)


    def fit_transform(self, x, y):
        self.fit(x,y)
        return self.transform(x)

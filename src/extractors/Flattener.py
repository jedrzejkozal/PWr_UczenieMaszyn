import numpy as np
from utils import flatten

class Flattener:
    def __init__(self, xTrain, yTrain):
        pass

    def transform(self, x):
        return flatten(x)

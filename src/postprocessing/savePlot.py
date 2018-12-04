#from postprocessing.utils import getFirstItemFromDict
from utils import getFirstItemFromDict

import matplotlib.pyplot as plt
import numpy as np

class SavePlot:

    def saveBarPlot(self, dataDict, fileName, labels, ylabel=None, title=None, setLogScale=False):
        numberOfGroups = len(getFirstItemFromDict(dataDict))
        numberOfBarsInGroup = len(dataDict.items())

        barWidth = 0.1
        groupsLocationX = np.arange(numberOfGroups)

        locationOfFirstBar = -(numberOfBarsInGroup-1)*barWidth/2
        locationOfLastBar = (numberOfBarsInGroup-1)*barWidth/2
        locationCorrection = np.arange(locationOfFirstBar,
            locationOfLastBar + barWidth, barWidth)

        fig, ax = plt.subplots()
        colors = ['g', 'r', 'y', 'b', 'm', 'c', 'cyan', 'tan', 'SkyBlue',
            'IndianRed', 'coral', 'pink']
        index = 0

        for classifierName, dict in dataDict.items():
            ax.bar(groupsLocationX + locationCorrection[index], tuple(dict),
                barWidth, color=colors[index], label=classifierName)
            index += 1

        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        ax.set_xticks(groupsLocationX)
        ax.set_xticklabels(labels)
        ax.legend()

        if setLogScale:
            plt.yscale("log")

        fullPath = '../doc/images/' + fileName + '.png'
        fig.savefig(fullPath)

        plt.close()

from postprocessing.saveTexTable import SaveTexTable
from postprocessing.savePlot import SavePlot
from postprocessing.anovaAnalysis import AnovaAnalysis
from postprocessing.utils import getFirstItemFromDict
#from saveTexTable import SaveTexTable
#from savePlot import SavePlot
#from anovaAnalysis import AnovaAnalysis

import numpy as np

class ProcessResults:

    def __init__(self):
        self.saveTable = SaveTexTable()
        self.savePlot = SavePlot()
        self.anova = AnovaAnalysis()


    def process(self, results, classifierName):
        labels = self.getExtractorsLabels(results)

        classifiersResult = self.getDictWith(results, self.avrgTestScoreSelector)
        self.savePlot.saveBarPlot(classifiersResult, classifierName+"_accuracy_comparison",
            labels, ylabel='Accuracy')

        self.numberOfExtractionMethods = len(getFirstItemFromDict(classifiersResult))
        self.numberOfClassifiers = len(classifiersResult.items())

        fitTimes = self.getDictWith(results, self.fitTimeSelector)
        self.savePlot.saveBarPlot(fitTimes, classifierName+"_fit_time_comparison",
            labels, ylabel='Traning time',
            setLogScale=True)

        self.saveTable.saveTable(results, self.avrgTestScoreSelector, classifierName+"_acc_table")
        self.saveTable.saveTable(results, self.fitTimeSelector, classifierName+"_fit_time_table")

        self.doStatisticalAnalysis(results, classifierName)


    def avrgTestScoreSelector(self, scores):
        return scores["test_score"].mean()


    def completeTestScoreSelector(self, scores):
        return scores["test_score"]


    def fitTimeSelector(self, scores):
        return scores["fit_time"].mean()


    def getDictWith(self, results, selector):
        result = {}

        for _, extractionResult in results.items():
            for classifierName, scores in extractionResult.items():
                if classifierName in result:
                    result[classifierName].append(selector(scores))
                else:
                    result[classifierName] = [selector(scores)]

        return result


    def getExtractorsLabels(self, results):
        labels = []

        for extractorName, _ in results.items():
            labels.append(extractorName)

        return labels


    def getClassifiersLabels(self, results):
        labels = []
        train_score = self.getDictWith(results, self.completeTestScoreSelector)

        for classifierName, _ in train_score.items():
            labels.append(classifierName)

        return labels


    def doStatisticalAnalysis(self, results):
        errorTables = self.getErrorTableForAllClassifiers(results)
        classifiersLabels = self.getClassifiersLabels(results)
        extractorLabels = self.getExtractorsLabels(results)

        self.saveTableWithPvalie(results, errorTables, classifiersLabels, extractorLabels, classifierName)


    def saveTableWithPvalie(self, results, errorTables, classifiersLabels, extractorLabels, classifierName):
        tableToSave = [["metoda ekstrakcji"], ["p-wartość"], ["wartość statystyki F"]]
        for name, _ in getFirstItemFromDict(results).items():
            tableToSave[0].append(name)

        for classifierName, errorTableForClassfier in zip(classifiersLabels, errorTables):
            statistic, pvalue = self.anova.test_null_hipotesis(classifierName,
                extractorLabels, errorTableForClassfier)
            tableToSave[1].append('{:7e}'.format(pvalue))
            tableToSave[2].append('{0:.2f}'.format(statistic))

        self.saveTable.saveTexTable(tableToSave, classifierName+"_pvalues")


    def getErrorTableForAllClassifiers(self, results):
        train_score = self.getDictWith(results, self.completeTestScoreSelector)
        errorTables = []

        L = self.numberOfClassifiers
        K = 10 #number of folds
        numberOfSamplesInFold = 206

        for classifierName, scoresForAllExtractionMethos in train_score.items():
            toAdd = []
            for i, extractionTrainScore in zip(range(self.numberOfExtractionMethods), scoresForAllExtractionMethos):
                elem = ((np.ones(extractionTrainScore.shape) - extractionTrainScore)*numberOfSamplesInFold).astype(int).tolist()
                toAdd.append(elem)
            errorTables.append(toAdd)

        return errorTables

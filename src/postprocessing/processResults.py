from postprocessing.saveTexTable import SaveTexTable
from postprocessing.savePlot import SavePlot
from postprocessing.statisticalAnalysis import StatisticalAnalysis
from postprocessing.utils import getFirstItemFromDict
from postprocessing.utils import saveResults
#from saveTexTable import SaveTexTable
#from savePlot import SavePlot
#from statisticalAnalysis import StatisticalAnalysis
#from utils import getFirstItemFromDict
#from utils import saveResults

import numpy as np

class ProcessResults:

    def __init__(self):
        self.saveTable = SaveTexTable()
        self.savePlot = SavePlot()
        self.statistical = StatisticalAnalysis()


    def process(self, results, classifierName):
        #print("results: ", results)
        saveResults(results, classifierName+"_results")

        labels = self.getDatabasesLabels(results)
        extractorsResult = self.getDictWith(results, self.avrgTestScoreSelector)
        self.savePlot.saveBarPlot(extractorsResult, classifierName+"_accuracy_comparison",
            labels, ylabel='Accuracy')

        self.numberOfDatabases = len(getFirstItemFromDict(extractorsResult))
        self.numberOfExtractors = len(extractorsResult.items())

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

        for _, databaseResult in results.items():
            for extractorName, scores in databaseResult.items():
                if extractorName in result:
                    result[extractorName].append(selector(scores))
                else:
                    result[extractorName] = [selector(scores)]

        return result


    def getDatabasesLabels(self, results):
        labels = []

        for databaseName, _ in results.items():
            labels.append(databaseName)

        return labels


    def getExtractorsLabels(self, results):
        labels = []
        train_score = self.getDictWith(results, self.completeTestScoreSelector)

        for extractorName, _ in train_score.items():
            labels.append(extractorName)

        return labels


    def doStatisticalAnalysis(self, results, classifierName):
        errorTables = self.getErrorTableForAllExtractors(results)
        extractorsLabels = self.getExtractorsLabels(results)

        statistic, pvalue, posthoc = self.statistical.testHypothesis(errorTables)

        self.saveTableWithPvalue(statistic, pvalue, classifierName+"_pvalues")
        saveResults(posthoc, classifierName+"_postHoc")


    def saveTableWithPvalue(self, statistic, pvalue, fileName):
        tableToSave = [[" ", "values"], ["p-value"], ["value of statistic F"]]

        tableToSave[1].append('{:7e}'.format(pvalue))
        tableToSave[2].append('{0:.2f}'.format(statistic))

        self.saveTable.saveTexTable(tableToSave, fileName)


    def getErrorTableForAllExtractors(self, results):
        train_score = self.getDictWith(results, self.avrgTestScoreSelector)
        errorTables = [[], [], []]

        for i, extractorTrainScore in zip(range(self.numberOfExtractors), train_score.items()):
            errorTables[i].append(extractorTrainScore[1])

        return errorTables

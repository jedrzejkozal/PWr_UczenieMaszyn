#from postprocessing.saveTexTable import SaveTexTable
#from postprocessing.savePlot import SavePlot
#from postprocessing.statiscalAnalysis import StatiscalAnalysis
#from postprocessing.utils import getFirstItemFromDict
from saveTexTable import SaveTexTable
from savePlot import SavePlot
from statiscalAnalysis import StatiscalAnalysis
from utils import getFirstItemFromDict

import numpy as np

class ProcessResults:

    def __init__(self):
        self.saveTable = SaveTexTable()
        self.savePlot = SavePlot()
        self.statiscal = StatiscalAnalysis()


    def process(self, results, classifierName):
        labels = self.getDatabasesLabels(results)

        print("results: ", results)

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

        self.saveTableWithPvalue(results, errorTables, extractorsLabels,
            classifierName)


    def saveTableWithPvalue(self, results, errorTables, extractorsLabels, classifierName):
        tableToSave = [[" ", "values"], ["p-value"], ["value of statistic F"]]

        statistic, pvalue = self.statiscal.testNullHypothesis(errorTables)
        tableToSave[1].append('{:7e}'.format(pvalue))
        tableToSave[2].append('{0:.2f}'.format(statistic))

        self.saveTable.saveTexTable(tableToSave, classifierName+"_pvalues")


    def getErrorTableForAllExtractors(self, results):
        train_score = self.getDictWith(results, self.avrgTestScoreSelector)
        errorTables = [[], [], []]

        for i, extractorTrainScore in zip(range(self.numberOfExtractors), train_score.items()):
            errorTables[i].append(extractorTrainScore[1])

        return errorTables




if __name__ == "__main__":

    arg = {'hands digits:': {'CNN': {'fit_time': np.array([41.09881115, 32.05074167, 40.12821174, 42.827425  , 43.51534581,
       38.6638999 , 42.60091305, 38.32311487, 40.71226764, 39.717659  ]), 'score_time': np.array([0.44809604, 0.59721804, 0.50908971, 0.36806273, 0.75289488,
       0.38465977, 0.61176682, 0.34503698, 0.50811362, 0.65724778]), 'test_score': np.array([0.8952381 , 0.9       , 0.91904762, 0.85714286, 0.86602871,
       0.93269231, 0.90196078, 0.87562189, 0.885     , 0.88      ]), 'train_score': np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])}, 'LDA': {'fit_time': np.array([6.58350778, 5.70512056, 5.68869352, 5.70475984, 5.81715679,
       5.80032015, 5.78563452, 5.84001446, 5.88697743, 5.86401606]), 'score_time': np.array([0.0052619 , 0.00517988, 0.00524569, 0.00523973, 0.00524855,
       0.00511932, 0.00509882, 0.00495934, 0.00492978, 0.00494719]), 'test_score': np.array([0.63809524, 0.6047619 , 0.58571429, 0.50952381, 0.60287081,
       0.57692308, 0.62254902, 0.59701493, 0.56      , 0.595     ]), 'train_score': np.array([0.97516199, 0.97894168, 0.97516199, 0.97786177, 0.97841338,
       0.9768069 , 0.98062433, 0.97796883, 0.97744361, 0.97529538])}, 'PCA': {'fit_time': np.array([5.14121652, 5.04868078, 5.06676483, 5.06719685, 5.05462623,
       5.08360696, 5.07999516, 5.08329153, 5.08469367, 5.11602664]), 'score_time': np.array([0.01104116, 0.01100206, 0.01036191, 0.01043105, 0.01050115,
       0.01023769, 0.01015496, 0.01010108, 0.0098238 , 0.00997376]), 'test_score': np.array([0.37142857, 0.34761905, 0.36666667, 0.36666667, 0.35406699,
       0.36538462, 0.41176471, 0.3681592 , 0.435     , 0.375     ]), 'train_score': np.array([0.37365011, 0.35475162, 0.32991361, 0.34611231, 0.39449541,
       0.37162891, 0.38266954, 0.42074154, 0.35875403, 0.41192266])}}, 'hands digits same:': {'CNN': {'fit_time': np.array([33.94790244, 37.9620018 , 34.52393293, 42.56917858, 40.71315074,
       36.42746091, 45.7413311 , 44.76957083, 36.27902246, 38.94157338]), 'score_time': np.array([0.44482422, 0.4382062 , 0.39355636, 0.6662004 , 0.35899305,
       0.42355323, 1.04408264, 0.38887858, 0.4203825 , 0.39659882]), 'test_score': np.array([0.9047619 , 0.9       , 0.88571429, 0.94761905, 0.85167464,
       0.90384615, 0.8627451 , 0.90049751, 0.91      , 0.9       ]), 'train_score': np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])}, 'LDA': {'fit_time': np.array([5.88594437, 5.67983699, 5.69046402, 5.71996331, 5.78156257,
       5.77269888, 5.84139442, 5.85778379, 5.87251925, 5.86611819]), 'score_time': np.array([0.00521326, 0.00525188, 0.00519419, 0.00538826, 0.00521612,
       0.00526094, 0.00511241, 0.00512552, 0.00510788, 0.00505757]), 'test_score': np.array([0.54285714, 0.62857143, 0.58095238, 0.65238095, 0.64593301,
       0.57692308, 0.56862745, 0.65174129, 0.595     , 0.58      ]), 'train_score': np.array([0.97462203, 0.97678186, 0.97516199, 0.98056156, 0.97409606,
       0.97626753, 0.97631862, 0.97581945, 0.97744361, 0.97798067])}, 'PCA': {'fit_time': np.array([5.0699141 , 5.06007361, 5.04641485, 5.04727387, 5.04307532,
       5.05647659, 5.06736684, 5.07166481, 5.08055162, 5.0645535 ]), 'score_time': np.array([0.01062465, 0.01065731, 0.01070285, 0.01064157, 0.01056099,
       0.01114988, 0.01044297, 0.0103066 , 0.01018929, 0.01022911]), 'test_score': np.array([0.36190476, 0.32380952, 0.38095238, 0.4       , 0.41626794,
       0.22115385, 0.29411765, 0.34328358, 0.37      , 0.37      ]), 'train_score': np.array([0.42278618, 0.37473002, 0.39470842, 0.37257019, 0.42093902,
       0.27562028, 0.37674919, 0.39226222, 0.32116004, 0.3915145 ])}}}



    p = ProcessResults()
    p.process(arg, "Svm")

from postprocessing.saveTexTable import SaveTexTable
from postprocessing.savePlot import SavePlot
from postprocessing.statisticalAnalysis import StatisticalAnalysis
from postprocessing.utils import getFirstItemFromDict
#from saveTexTable import SaveTexTable
#from savePlot import SavePlot
#from statisticalAnalysis import StatisticalAnalysis
#from utils import getFirstItemFromDict

import numpy as np

class ProcessResults:

    def __init__(self):
        self.saveTable = SaveTexTable()
        self.savePlot = SavePlot()
        self.statistical = StatisticalAnalysis()


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
        self.saveResults(results, classifierName+"_results")


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
        self.saveResults(posthoc, classifierName+"_postHoc")


    def saveTableWithPvalue(self, statistic, pvalue, fileName):
        tableToSave = [[" ", "values"], ["p-value"], ["value of statistic F"]]

        tableToSave[1].append('{:7e}'.format(pvalue))
        tableToSave[2].append('{0:.2f}'.format(statistic))

        self.saveTable.saveTexTable(tableToSave, fileName)


    def saveResults(self, results, fileName):
        path = "../doc/tables/" + fileName + ".txt"
        f = open(path, 'w')
        f.write(str(results))
        f.close()


    def getErrorTableForAllExtractors(self, results):
        train_score = self.getDictWith(results, self.avrgTestScoreSelector)
        errorTables = [[], [], []]

        for i, extractorTrainScore in zip(range(self.numberOfExtractors), train_score.items()):
            errorTables[i].append(extractorTrainScore[1])

        return errorTables




if __name__ == "__main__":

    arg = {'att:': {'CNN': {'fit_time': np.array([ 9.73712945,  7.11988163,  8.05349183,  7.94410205, 10.29620171,
       10.56885171,  8.97548914, 10.14933944,  8.59427857,  9.69579959]), 'score_time': np.array([0.09158444, 0.0864749 , 0.08191729, 0.08450031, 0.08302522,
       0.08150887, 0.27435803, 0.27574182, 0.07171535, 0.26144338]), 'test_score': np.array([1.   , 0.95 , 1.   , 1.   , 0.975, 0.925, 0.95 , 0.975, 0.975,
       1.   ]), 'train_score': np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])}, 'LDA': {'fit_time': np.array([0.69652343, 0.56391597, 0.5694797 , 0.56454897, 0.56236124,
       0.56230187, 0.56459379, 0.56062531, 0.56395602, 0.57096577]), 'score_time': np.array([0.00168204, 0.00166893, 0.00171185, 0.00165248, 0.00167108,
       0.00168705, 0.00165629, 0.00167441, 0.001647  , 0.00165868]), 'test_score': np.array([1.   , 0.975, 1.   , 0.925, 0.95 , 0.875, 0.95 , 0.95 , 0.975,
       0.95 ]), 'train_score': np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])}, 'PCA': {'fit_time': np.array([3.0377779 , 3.03450751, 3.03387666, 3.03574753, 3.0358336 ,
       3.02624106, 3.0319159 , 3.03152728, 3.03685355, 3.0300951 ]), 'score_time': np.array([0.00364947, 0.00366664, 0.00361872, 0.00364661, 0.00373149,
       0.00364399, 0.00362515, 0.0036459 , 0.00374532, 0.00359917]), 'test_score': np.array([0.55 , 0.625, 0.475, 0.35 , 0.475, 0.45 , 0.5  , 0.55 , 0.525,
       0.575]), 'train_score': np.array([0.64722222, 0.66388889, 0.66944444, 0.65277778, 0.69444444,
       0.65555556, 0.60277778, 0.68333333, 0.72777778, 0.71666667])}}, 'hands digits:': {'CNN': {'fit_time': np.array([33.02510095, 46.05610943, 43.79078364, 54.94672513, 52.56463575,
       56.41898513, 54.89646411, 55.55943251, 51.74866319, 46.64121532]), 'score_time': np.array([0.52122259, 0.60467052, 0.50906849, 0.62468147, 0.78036427,
       0.80703616, 0.94783282, 0.68016624, 0.75318456, 0.45301223]), 'test_score': np.array([0.9047619 , 0.91904762, 0.9       , 0.88095238, 0.89952153,
       0.88942308, 0.88235294, 0.87064677, 0.88      , 0.895     ]), 'train_score': np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])}, 'LDA': {'fit_time': np.array([6.06999755, 5.83840847, 5.8305614 , 5.78747988, 5.90850353,
       5.86860633, 5.93162346, 6.032969  , 6.01143312, 6.02672362]), 'score_time': np.array([0.00522661, 0.00536585, 0.00522113, 0.00542188, 0.00526857,
       0.00535822, 0.00524664, 0.00561333, 0.00530028, 0.00552201]), 'test_score': np.array([0.64761905, 0.62857143, 0.58571429, 0.57619048, 0.60287081,
       0.61057692, 0.55882353, 0.53731343, 0.55      , 0.575     ]), 'train_score': np.array([0.97732181, 0.97732181, 0.97732181, 0.97732181, 0.97625472,
       0.97572816, 0.97416577, 0.98119291, 0.97851772, 0.97744361])}, 'PCA': {'fit_time': np.array([5.34311342, 5.64445996, 5.28642225, 5.2062459 , 5.32895637,
       5.26380682, 5.1891582 , 5.19789886, 5.28516626, 5.26967335]), 'score_time': np.array([0.01036787, 0.01180029, 0.01119637, 0.01042628, 0.01090074,
       0.01047945, 0.01024008, 0.01039004, 0.01020217, 0.01082087]), 'test_score': np.array([0.32380952, 0.32380952, 0.36666667, 0.4       , 0.3923445 ,
       0.39423077, 0.28921569, 0.33830846, 0.37      , 0.355     ]), 'train_score': np.array([0.35961123, 0.37796976, 0.36933045, 0.45464363, 0.41122504,
       0.38781014, 0.36544672, 0.34390113, 0.39581096, 0.39742213])}}}



    p = ProcessResults()
    p.process(arg, "Svm")

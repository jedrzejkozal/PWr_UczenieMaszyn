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


    def saveTableWithPvalue(self, results, errorTables, extractorsLabels, extractorName):
        tableToSave = [["feature extraction method"], ["p-value"], ["value of statistic F"]]
        for name, _ in getFirstItemFromDict(results).items():
            tableToSave[0].append(name)

        for extractorName, errorTableForExtractor in zip(extractorsLabels, errorTables):
            statistic, pvalue = self.statiscal.testNullHypothesis(extractorName,
                errorTableForExtractor)
            tableToSave[1].append('{:7e}'.format(pvalue))
            tableToSave[2].append('{0:.2f}'.format(statistic))

        self.saveTable.saveTexTable(tableToSave, extractorName+"_pvalues")


    def getErrorTableForAllExtractors(self, results):
        train_score = self.getDictWith(results, self.completeTestScoreSelector)
        errorTables = []

        for extractorName, scoresForDatabases in train_score.items():
            toAdd = []
            for i, databaseTrainScore in zip(range(self.numberOfDatabases), scoresForDatabases):
                toAdd.append(databaseTrainScore.tolist())
            errorTables.append(toAdd)

        return errorTables




if __name__ == "__main__":
    arg = {'hands digits:': {'LDA': {'fit_time': np.array([4.84076834, 4.69979095, 4.8373127 , 5.04802823, 4.90385699,
       4.79904532, 5.28576994, 5.11618185, 5.07327867, 5.03035331]), 'score_time': np.array([0.00514603, 0.00500011, 0.00509238, 0.00497437, 0.00490928,
       0.00491714, 0.0063436 , 0.004807  , 0.00476003, 0.00519109]), 'test_score': np.array([0.54761905, 0.55238095, 0.57142857, 0.6       , 0.61722488,
       0.58653846, 0.55882353, 0.58208955, 0.59      , 0.585     ]), 'train_score': np.array([0.97516199, 0.97516199, 0.97840173, 0.97462203, 0.97517539,
       0.9768069 , 0.97631862, 0.97581945, 0.97583244, 0.97690655])}, 'PCA': {'fit_time': np.array([4.95297956, 4.87015963, 5.01747727, 4.94189501, 5.29439402,
       5.00581288, 5.21516919, 5.08671165, 5.05369806, 4.91486049]), 'score_time': np.array([0.01190424, 0.00875854, 0.01200867, 0.01245308, 0.00860095,
       0.00906658, 0.00920987, 0.00853372, 0.00841236, 0.00841546]), 'test_score': np.array([0.34761905, 0.38571429, 0.38095238, 0.4047619 , 0.38755981,
       0.37019231, 0.35784314, 0.33830846, 0.315     , 0.395     ]), 'train_score': np.array([0.4087473 , 0.38552916, 0.39632829, 0.42980562, 0.39719374,
       0.39536138, 0.39558665, 0.38259001, 0.39742213, 0.41245972])}}, 'hands digits same:': {'LDA': {'fit_time': np.array([4.70766187, 4.69797611, 4.79271507, 4.74328542, 4.77852631,
       4.76960659, 4.80923915, 4.76882744, 4.92569876, 4.80310774]), 'score_time': np.array([0.00496817, 0.00506234, 0.00505853, 0.00498676, 0.00503922,
       0.0079689 , 0.00485849, 0.00485182, 0.00487876, 0.00495815]), 'test_score': np.array([0.57142857, 0.56190476, 0.58095238, 0.56190476, 0.57894737,
       0.55288462, 0.57843137, 0.56218905, 0.6       , 0.58      ]), 'train_score': np.array([0.97516199, 0.97570194, 0.97948164, 0.97678186, 0.97625472,
       0.97626753, 0.97416577, 0.97850618, 0.97583244, 0.97475832])}, 'PCA': {'fit_time': np.array([4.77007341, 4.79104519, 4.73034263, 4.84283543, 4.63828349,
       4.63713813, 4.66034746, 4.6785183 , 4.67736244, 4.67950535]), 'score_time': np.array([0.00867462, 0.00858235, 0.00868893, 0.00869107, 0.00841784,
       0.00849462, 0.00834131, 0.00824142, 0.00819874, 0.00832367]), 'test_score': np.array([0.33809524, 0.34285714, 0.35238095, 0.33333333, 0.30143541,
       0.35576923, 0.36764706, 0.37313433, 0.345     , 0.435     ]), 'train_score': np.array([0.35043197, 0.37850972, 0.34773218, 0.36663067, 0.34700486,
       0.3425027 , 0.36006459, 0.40300913, 0.38238453, 0.41460795])}}}


    p = ProcessResults()
    p.process(arg, "Svm")

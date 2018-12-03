from postprocessing.saveTexTable import SaveTexTable
from postprocessing.savePlot import SavePlot
from postprocessing.anovaAnalysis import AnovaAnalysis
#from saveTexTable import SaveTexTable
#from savePlot import SavePlot
#from anovaAnalysis import AnovaAnalysis

import numpy as np

class ProcessResults:

    def __init__(self):
        self.saveTable = SaveTexTable()
        self.savePlot = SavePlot()
        self.anova = AnovaAnalysis()


    def process(self, results):
        labels = self.getExtractorsLabels(results)

        classifiersResult = self.getDictWith(results, self.avrgTestScoreSelector)
        self.savePlot.saveBarPlot(classifiersResult, "accuracy_comparison", labels, ylabel='Dokładność')

        self.numberOfExtractionMethods = len(classifiersResult["Svm"])
        self.numberOfClassifiers = len(classifiersResult.items())

        fitTimes = self.getDictWith(results, self.fitTimeSelector)
        self.savePlot.saveBarPlot(fitTimes, "fit_time_comparison", labels, ylabel='Czas trenowania',
            setLogScale=True)

        self.saveTable.saveTable(results, self.avrgTestScoreSelector, "acc_table")
        self.saveTable.saveTable(results, self.fitTimeSelector, "fit_time_table")

        self.doStatisticalAnalysis(results)


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

        self.saveTableWithPvalie(results, errorTables, classifiersLabels, extractorLabels)


    def saveTableWithPvalie(self, results, errorTables, classifiersLabels, extractorLabels):
        tableToSave = [["metoda ekstrakcji"], ["p-wartość"], ["wartość statystyki F"]]
        for name, _ in results["PCA"].items():
            tableToSave[0].append(name)

        for classifierName, errorTableForClassfier in zip(classifiersLabels, errorTables):
            statistic, pvalue = self.anova.test_null_hipotesis(classifierName,
                extractorLabels, errorTableForClassfier)
            tableToSave[1].append('{:7e}'.format(pvalue))
            tableToSave[2].append('{0:.2f}'.format(statistic))

        self.saveTable.saveTexTable(tableToSave, "pvalues")


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




if __name__ == "__main__":

    arg = {'PCA': {'KNN': {'fit_time': np.array([0.00968719, 0.0080564 , 0.0080688 , 0.00797415, 0.00798082,
       0.00798798, 0.00797248, 0.0079987 , 0.00809264, 0.00801897]), 'score_time': np.array([0.11342382, 0.11434889, 0.11300707, 0.10947371, 0.11200714,
       0.10977364, 0.11077929, 0.11133409, 0.11019111, 0.11045718]), 'test_score': np.array([0.69411765, 0.63905325, 0.62874251, 0.66666667, 0.70121951,
       0.68292683, 0.70731707, 0.69512195, 0.67901235, 0.75776398]), 'train_score': np.array([0.81756757, 0.81498987, 0.81861092, 0.83434343, 0.81763122,
       0.83109017, 0.82907133, 0.81763122, 0.8219086 , 0.82740094])}, 'NaiveBayes': {'fit_time': np.array([0.00779319, 0.00751185, 0.00731277, 0.00751424, 0.00730729,
       0.00734282, 0.00748038, 0.00731277, 0.00729465, 0.00743866]), 'score_time': np.array([0.00356483, 0.00325227, 0.00312686, 0.00316954, 0.00313854,
       0.00308084, 0.00316453, 0.00308681, 0.00306582, 0.00312424]), 'test_score': np.array([0.66470588, 0.63313609, 0.66467066, 0.63636364, 0.69512195,
       0.63414634, 0.58536585, 0.67073171, 0.65432099, 0.65838509]), 'train_score': np.array([0.81959459, 0.82309251, 0.83142279, 0.82828283, 0.83310902,
       0.83580081, 0.82234186, 0.82368775, 0.82526882, 0.83008731])}, 'RandomForest': {'fit_time': np.array([0.19384027, 0.18281579, 0.19550157, 0.18748212, 0.18336797,
       0.18250608, 0.19075036, 0.18655276, 0.18812203, 0.1872735 ]), 'score_time': np.array([0.00224757, 0.00223827, 0.00222588, 0.00221658, 0.00217295,
       0.00218797, 0.00220132, 0.00217748, 0.00218844, 0.002177  ]), 'test_score': np.array([0.36470588, 0.39053254, 0.34131737, 0.38787879, 0.40243902,
       0.31707317, 0.25      , 0.34146341, 0.37037037, 0.41614907]), 'train_score': np.array([0.99324324, 0.99594868, 0.99460553, 0.99461279, 0.99461642,
       0.99461642, 0.99528937, 0.9986541 , 0.99663978, 0.99261249])}}, 'PCA with LBP': {'KNN': {'fit_time': np.array([0.01293945, 0.01088142, 0.01083732, 0.01108479, 0.0109129 ,
       0.01089716, 0.01103973, 0.01085806, 0.01093102, 0.0110178 ]), 'score_time': np.array([0.17147684, 0.17231917, 0.16969657, 0.16598153, 0.16726708,
       0.1670866 , 0.16702414, 0.16674972, 0.16383004, 0.1651361 ]), 'test_score': np.array([0.59411765, 0.66272189, 0.64670659, 0.61212121, 0.73170732,
       0.64634146, 0.70731707, 0.68292683, 0.64197531, 0.68322981]), 'train_score': np.array([0.80675676, 0.80081026, 0.80107889, 0.80808081, 0.80686406,
       0.80753701, 0.80417227, 0.80686406, 0.81653226, 0.80456682])}, 'NaiveBayes': {'fit_time': np.array([0.01051521, 0.00996041, 0.009866  , 0.00977707, 0.00977349,
       0.00978661, 0.00982666, 0.00999451, 0.00987124, 0.00995779]), 'score_time': np.array([0.0042274 , 0.00423408, 0.00410175, 0.00416398, 0.00404572,
       0.00448346, 0.00405574, 0.00404716, 0.00405455, 0.00400996]), 'test_score': np.array([0.66470588, 0.67455621, 0.62874251, 0.61212121, 0.67073171,
       0.65243902, 0.65853659, 0.68292683, 0.63580247, 0.67701863]), 'train_score': np.array([0.73783784, 0.73396354, 0.73904248, 0.73602694, 0.72678331,
       0.73687752, 0.73149394, 0.71870794, 0.7405914 , 0.73069174])}, 'RandomForest': {'fit_time': np.array([0.21763515, 0.21272755, 0.20606971, 0.21277308, 0.21760917,
       0.2097652 , 0.21282101, 0.21377826, 0.22098899, 0.21546745]), 'score_time': np.array([0.00239491, 0.00228119, 0.00225902, 0.00239253, 0.00225139,
       0.00245786, 0.0023675 , 0.00226903, 0.00226665, 0.00238013]), 'test_score': np.array([0.47058824, 0.43195266, 0.38922156, 0.46666667, 0.45121951,
       0.48170732, 0.48780488, 0.43292683, 0.43209877, 0.44720497]), 'train_score': np.array([0.9972973 , 0.99324781, 0.99797707, 0.99461279, 0.99730821,
       0.99596231, 0.99528937, 0.99461642, 0.99596774, 0.99529886])}}, 'Skeletonize': {'KNN': {'fit_time': np.array([0.02523923, 0.02404642, 0.02382016, 0.02381349, 0.02406764,
       0.02396703, 0.02401781, 0.02386189, 0.02382421, 0.02394748]), 'score_time': np.array([0.31307387, 0.31047225, 0.30740094, 0.30311131, 0.30443645,
       0.30214047, 0.30259013, 0.30279326, 0.29862142, 0.2986331 ]), 'test_score': np.array([0.55294118, 0.5739645 , 0.60479042, 0.54545455, 0.59756098,
       0.5       , 0.59146341, 0.56707317, 0.61728395, 0.60248447]), 'train_score': np.array([0.71418919, 0.72113437, 0.72825354, 0.72525253, 0.73889637,
       0.72341857, 0.73149394, 0.73216689, 0.72715054, 0.72196105])}, 'NaiveBayes': {'fit_time': np.array([0.01865077, 0.01486683, 0.01521802, 0.01571035, 0.01508522,
       0.01482081, 0.01497293, 0.01490545, 0.0154326 , 0.01495671]), 'score_time': np.array([0.00765657, 0.00625992, 0.00611281, 0.00617146, 0.00604653,
       0.0059762 , 0.00636744, 0.00608587, 0.00599027, 0.00597596]), 'test_score': np.array([0.67058824, 0.66272189, 0.73053892, 0.65454545, 0.57317073,
       0.56707317, 0.62804878, 0.6402439 , 0.70987654, 0.66459627]), 'train_score': np.array([0.80405405, 0.75286968, 0.76803776, 0.74478114, 0.74764468,
       0.75841184, 0.75841184, 0.76514132, 0.76276882, 0.75352586])}, 'RandomForest': {'fit_time': np.array([0.06118035, 0.0665102 , 0.0669384 , 0.06444669, 0.06543732,
       0.06452489, 0.06570935, 0.064533  , 0.06400084, 0.06574678]), 'score_time': np.array([0.00237107, 0.00221944, 0.00233197, 0.00228906, 0.00231194,
       0.00228381, 0.00231409, 0.00234342, 0.00243855, 0.00230813]), 'test_score': np.array([0.43529412, 0.56213018, 0.49101796, 0.44848485, 0.48170732,
       0.51219512, 0.51829268, 0.52439024, 0.47530864, 0.50310559]), 'train_score': np.array([0.99324324, 0.99392302, 0.99527984, 0.996633  , 0.99596231,
       0.99798116, 0.99663526, 0.99596231, 0.99462366, 0.99529886])}}}

    p = ProcessResults()
    p.doStatisticalAnalysis(arg)

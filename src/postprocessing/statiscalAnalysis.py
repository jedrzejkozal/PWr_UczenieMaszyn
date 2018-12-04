import scipy.stats as stats
import numpy as np
#from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp

class StatiscalAnalysis:

    def __init__(self):
        pass


    def test_null_hipotesis(self, classifierName, extractorLabels, errorTable):

        #please forgive me, stats.f_oneway dosn't accept list of lists or tuple
        a = errorTable[0]
        b = errorTable[1]
        c = errorTable[0]

        statistic, pvalue = stats.friedmanchisquare(a, b, c)

        print("="*40)
        print(classifierName)
        print("statistic: ", statistic)
        print("pvalue: ", pvalue)

        if self.evaluateHypothesis(pvalue):
            print("Post hoc testing skipped due to not rejected ANOVA H0 hypothesis (u1 = u2 = u3)")
        else:
            self.doPostHocTesting(errorTable, extractorLabels)
        self.doPostHocTesting(errorTable, extractorLabels) # to remove
        print("="*40)

        return statistic, pvalue


    def evaluateHypothesis(self, pvalue):
        if pvalue < 0.04:
            print("Rejecting H0: no evidence to support hypothesis")
            return False
        elif pvalue > 0.06:
            print("Fail to reject H0: strong indication, that hypothesis is in line with data")
            return True
        else:
            print("No conclusive result")
        return False


    def doPostHocTesting(self, errorTable, extractorLabels):
        print("errorTable: ", errorTable)
        res = sp.posthoc_nemenyi_friedman(errorTable)
        print("res:")
        print(res)


    """
    def convertErrorTableToSingleVector(self, errorTable, extractorLabels):
        errorTable = np.array(errorTable)
        result = errorTable.flatten()

        shape = errorTable.shape[0]*errorTable.shape[1]
        labels = []
        for i, label in zip(range(len(extractorLabels)), extractorLabels):
            for j in range(i*errorTable.shape[1], (i+1)*errorTable.shape[1]):
                labels.append(label)

        labels = np.array(labels)

        return result, labels
    """

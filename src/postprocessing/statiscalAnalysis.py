import scipy.stats as stats
import numpy as np
import scikit_posthocs as sp

class StatiscalAnalysis:

    def __init__(self):
        pass


    def testNullHypothesis(self, errorTable):

        #please forgive me, stats.f_oneway dosn't accept list of lists or tuple
        a = errorTable[0]
        b = errorTable[1]
        c = errorTable[2]

        statistic, pvalue = stats.friedmanchisquare(a, b, c)
        self.printNullHypothesisResults(statistic, pvalue)
        self.evaluate(pvalue, errorTable)
        print("="*40)

        return statistic, pvalue


    def printNullHypothesisResults(self, statistic, pvalue):
        print("="*40)
        print("Comparison of extractors over multiple datasets result")
        print("statistic: ", statistic)
        print("pvalue: ", pvalue)


    def evaluate(self, pvalue, errorTable):
        if self.evaluateH0Hypothesis(pvalue):
            print("Post hoc testing skipped due to not rejected ANOVA H0 hypothesis (u1 = u2 = u3)")
        else:
            self.doPostHocTesting(errorTable)
        self.doPostHocTesting(errorTable) # to remove


    def evaluateH0Hypothesis(self, pvalue):
        if pvalue < 0.04:
            print("Rejecting H0: no evidence to support hypothesis")
            return False
        elif pvalue > 0.06:
            print("Fail to reject H0: strong indication, that hypothesis is in line with data")
            return True
        else:
            print("No conclusive result")
        return False


    def doPostHocTesting(self, errorTable):
        x = np.array(errorTable)
        x = x[:, 0, :]
        res = sp.posthoc_nemenyi_friedman(x)
        print("res:")
        print(res)

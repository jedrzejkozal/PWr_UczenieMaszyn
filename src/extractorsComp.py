from extractors import *

from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline


class ExtractorComparison:

    def __init__(self):
        self.extractorsDict = {
            'LDA': Lda.Lda,
            'PCA': Pca.Pca,
        }


    def compareExtractors(self, classifier, data, target):
        results = {}
        for name, extractionMethod in self.extractorsDict.items():
            print(name)
            extractor = extractionMethod(data, target)

            clf = make_pipeline(extractor, classifier)
            result = self.crossValidate(clf, data, target)
            results[name] = result
        return results


    def crossValidate(self, classifier, data, target):
        K = 10
        scores = cross_validate(classifier, data, target, cv=K)
        print(scores["test_score"].mean())
        return scores

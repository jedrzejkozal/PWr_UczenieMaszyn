from Db import Db
from extractors import *
from classifiers import ClassifiersComparison
from postprocessing.processResults import ProcessResults

from sklearn.model_selection import train_test_split


extractorsDict = {
    'LDA': Lda.Lda,
    'PCA': Pca.Pca,
}

if __name__ == "__main__":

    x, y = Db().getGrayscale()

    classifiersComparision = ClassifiersComparison()
    processResults = ProcessResults()

    results = {}

    for name, extractionMethod in extractorsDict.items():
        print("="*40)
        print("Extractor: {}".format(name))

        extractor = extractionMethod(x, y)

        result = classifiersComparision.compareClassifiers(extractor, x, y)
        results[name] = result

    processResults.process(results)

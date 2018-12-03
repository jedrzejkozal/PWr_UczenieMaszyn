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

    xTrainExtrator, x, yTrainExtrator, y = \
        train_test_split(x, y, test_size=0.8, random_state=42)

    classifiersComparision = ClassifiersComparison()
    processResults = ProcessResults()

    results = {}

    for name, extractionMethod in extractorsDict.items():
        print("="*40)
        print("Extractor: {}".format(name))

        if name == 'RPCA':
            extractor = extractionMethod(x, y)
        else:
            extractor = extractionMethod(xTrainExtrator, yTrainExtrator)

        xExtracted = extractor.transform(x)

        result = classifiersComparision.compareClassifiers(xExtracted, y)
        results[name] = result

    processResults.process(results)

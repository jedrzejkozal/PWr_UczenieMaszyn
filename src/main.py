from Db import Db
from extractorsComp import ExtractorComparison
from postprocessing.processResults import ProcessResults

from sklearn import svm
from sklearn.neural_network import MLPClassifier


classifiersDict = {
    #'NN': MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(300, 200, 150), random_state=1),
    'Svm': svm.LinearSVC()
}

dataBaseDict = {
    #'att:': Db("att", 40), #works
    #'hands digits:': Db("hands", 10), # works
    'caltec:': Db("croped_caltec", 19), #works
    #'georgia': Db("georgia", 50),
}

if __name__ == "__main__":

    extractorComparision = ExtractorComparison()
    processResults = ProcessResults()

    for classifierName, classifier in classifiersDict.items():

        print("="*40)
        print("="*40)
        print("classifier: {}".format(classifierName))
        print("="*40)
        print("="*40)

        results = {}


        for dbName, db in dataBaseDict.items():
            print("="*40)
            print("database: {}".format(dbName))

            x, y = db.getGrayscale()

            result = extractorComparision.compareExtractors(classifier, x, y,
                db.numClasses)
            results[dbName] = result

        processResults.process(results, classifierName)

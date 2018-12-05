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
    #'att:': Db("att", 40), #loads normaly
    #'hands digits:': Db("hands", 10), # loads normaly
    #'caltec:': Db("croped_caltec", 19), # loads normaly
    #'essex:': Db("essex", 392), # loads normaly, note: on my 8Gb RAM essex causes MemoryError
    #'georgia': Db("georgia", 50), #loads normaly
    #'jaffe': Db("jaffe", 10), # loads normaly
    #'mit': Db("mit-cbcl", 29), # loads normaly
    #'muct': Db("muct", 276), # loads normaly
    #'specs-on-faces': Db("specs-on-faces", 101), # loads normaly
    #'stirling': Db("stirling", 36), # loads normaly, but getting error: ValueError: n_splits=10 cannot be greater than the number of members in each class.
    'umist': Db("umist", 20),
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

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
    'essex:': Db("essex", 392), # loads normaly, note: on my 8Gb RAM essex causes MemoryError
    'vidtimit': Db("vidtimit", 14), # loads normaly, process gets killed, no stacktrace, only "Killed" apears in console, probably due to SIGKILL
    #'att:': Db("att", 40), #loads normaly, results checked
    #'caltec:': Db("croped_caltec", 19), # loads normaly, results checked
    #'georgia': Db("georgia", 50), #loads normaly, results checked
    #'jaffe': Db("jaffe", 10), # loads normaly, results checked
    #'mit': Db("mit-cbcl", 29), # loads normaly, results checked
    #'muct': Db("muct", 276), # loads normaly, results checked
    #'specs-on-faces': Db("specs-on-faces", 101), # loads normaly, results checked
    #'stirling': Db("stirling", 36), # loads normaly, but getting error: ValueError: n_splits=10 cannot be greater than the number of members in each class.
    #'umist': Db("umist", 20), # loads normaly, results checked
    #'yale': Db("yale", 15), # loads normaly, results checked
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

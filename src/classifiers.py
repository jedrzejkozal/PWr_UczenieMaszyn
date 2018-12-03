from sklearn import svm
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline


class ClassifiersComparison:

    def __init__(self):
        self.classifiersDict = {
            'NN': MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(300, 200, 150), random_state=1),
            'Svm': svm.LinearSVC()
        }


    def compareClassifiers(self, extractor, data, target):
        results = {}
        for name, classifier in self.classifiersDict.items():
            print(name)
            clf = make_pipeline(extractor, classifier)
            result = self.crossValidate(clf, data, target)
            results[name] = result
        return results


    def crossValidate(self, classifier, data, target):
        K = 10
        scores = cross_validate(classifier, data, target, cv=K)
        print(scores["test_score"].mean())
        return scores

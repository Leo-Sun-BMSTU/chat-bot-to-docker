from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, roc_auc_score, \
    f1_score, fbeta_score, auc
from sklearn.neighbors import KNeighborsClassifier


class Classifier:
    lab = property()
    clf = property()
    generalMetric = property()
    _metrics = {"accuracy": None, 'precision': None, 'recall': None, 'roc': None,
                'prc': None, 'fe1': None, 'f0.5': None, 'f2': None, }

 # def __init__(self):
 #     self._number_lab = None

    def get_clf_from_pickle(self, pickle):
        # while self._lab is None:
        try:
            pipeline = [v for v in pickle.get_params().values() if isinstance(v, LogisticRegression)]
            if pipeline[0].get_params().get('penalty') == 'l1':
                print("Classifier is not LogisticRegression L2")
                self._lab = "lab3_l1"
                self._clf = pipeline[0]
                return self._clf, self._lab
            else:
                if pipeline[0].get_params().get('penalty') == 'l2':
                    print("Classifier is not LogisticRegression L1")
                    self._lab = "lab3_l2"
                    self._clf = pipeline[0]
                    return self._clf, self._lab
        except IndexError:
            print("Classifier is not LogisticRegression")
        try:
            pipeline = [v for v in pickle.get_params().values() if isinstance(v, RandomForestClassifier)]
            self._lab = "lab4_RF"
            # self._number_lab = "lab4_RF"
            self._clf = pipeline[0]
            return self._clf, self._lab
        except IndexError:
            print("Classifier is not RandomForest")
        try:
            pipeline = [v for v in pickle.get_params().values() if isinstance(v, KNeighborsClassifier)]
            self._lab = "lab2"
            self._clf = pipeline[0]
            return self._clf, self._lab
        except IndexError:
            print("Classifier is not KNeighbours")


    def get_general_metric(self, metric):
        self._generalMetric = self._metrics.get(metric.lower())
        print("General Metric {} = {}".format(metric, self._generalMetric))


    def calculate_metrics(self, y_test, y_predict):
        self._metrics['accuracy'] = accuracy_score(y_test, y_predict)
        self._metrics['precision'] = precision_score(y_test, y_predict)
        self._metrics['recall'] = recall_score(y_test, y_predict)
        pr, rec, _ = precision_recall_curve(y_test, y_predict)
        self._metrics['roc'] = roc_auc_score(y_test, y_predict)
        self._metrics['prc'] = auc(pr, rec)
        self._metrics['f1'] = f1_score(y_test, y_predict)
        self._metrics['f0.5'] = fbeta_score(y_test, y_predict, 0.5)
        self._metrics['f2'] = fbeta_score(y_test, y_predict, 2)


    @clf.getter
    def clf(self):
        return self._clf
    @clf.setter
    def clf(self, value):
        self._clf = value
    @clf.deleter
    def clf(self):
        del self._clf

    @lab.getter
    def lab(self):
        return self._lab

    @lab.setter
    def lab(self, value):
        self._lab = value

    @lab.deleter
    def lab(self):
        del self._lab

    @generalMetric.getter
    def generalMetric(self):
        return self._generalMetric


    @generalMetric.setter
    def generalMetric(self, value):
        self._generalMetric = value


    @clf.deleter
    def generalMetric(self):
        del self._generalMetric

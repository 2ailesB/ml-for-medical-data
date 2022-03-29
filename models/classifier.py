from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.multiclass import OneVsRestClassifier
import sklearn.metrics as metrics
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

import pickle
from utils import format_paramsdict
from .NN import LSTM_classifier2
# setup plot details
colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

class Model(object):
    def __init__(self, train_data, test_data, preprocess, metric='f1_micro'):
        self.train_X = self.preprocess(train_data.iloc[:, 0:64])
        self.train_y = train_data.iloc[:, 64]

        self.test_X = test_data.iloc[:, 0:64]
        self.test_y = test_data.iloc[:, 64]

        self.preprocess = preprocess
        self.parameter_optimal = {}
        self.model = None
        self.metric = metric

    
    def grid_search(self, parameters, n_fold):
        """Grid search on train dataset"""
        # Parameters of pipelines can be set using ‘__’ separated parameter names:
        search = GridSearchCV(self.pipe, parameters, n_jobs=-1, cv=n_fold, scoring=self.metric)
        search.fit(self.train_X, self.train_y)
        self.model = search.best_estimator_
        self.search = search
        return self.search


    def visualisation(self, path, metrics='mean_test_score'):
        'save figure of cv'

        fig = plt.figure()
        scores = [x for x in self.gs_result.cv_results_[metrics]]
        plt.plot(scores)
        plt.xlabel('parameters')
        plt.ylabel(metrics)
        plt.title(f'{metrics} for models tested during grid search')
        plt.savefig(path)
        plt.clf()
        return None


    def evaluation(self, name):
        """Result evaluatio on test dataset with optimal parameters after grids search"""
        train_Y = label_binarize(self.train_y, classes=[0, 1, 2, 3])
        test_Y = label_binarize(self.test_y, classes=[0, 1, 2, 3])
        n_classes = test_Y.shape[1]
        
        classifier = OneVsRestClassifier(
            make_pipeline(StandardScaler(), self.model)
        )
        classifier.fit(self.train_X, train_Y)
        y_score = classifier.predict_proba(self.test_X)

        # For each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(test_Y[:, i], y_score[:, i])
            average_precision[i] = average_precision_score(test_Y[:, i], y_score[:, i])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            test_Y.ravel(), y_score.ravel()
        )
        average_precision["micro"] = average_precision_score(test_Y, y_score, average="micro")
        self.show_test_pr_curves(recall, precision, average_precision, n_classes, name)


    def show_test_pr_curves(self, recall, precision, average_precision, n_classes, name):
        """ ref: 
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py
        """
        _, ax = plt.subplots(figsize=(7, 8))

        f_scores = np.linspace(0.2, 0.8, num=4)
        lines, labels = [], []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
            plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

        display = PrecisionRecallDisplay(
            recall=recall["micro"],
            precision=precision["micro"],
            average_precision=average_precision["micro"],
        )
        display.plot(ax=ax, name="Micro-average precision-recall", color="gold")

        for i, color in zip(range(n_classes), colors):
            display = PrecisionRecallDisplay(
                recall=recall[i],
                precision=precision[i],
                average_precision=average_precision[i],
            )
            display.plot(ax=ax, name=f"Precision-recall for class {i}", color=color)

        # add the legend for the iso-f1 curves
        handles, labels = display.ax_.get_legend_handles_labels()
        handles.extend([l])
        labels.extend(["iso-f1 curves"])
        # set the legend and the axes
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(handles=handles, labels=labels, loc="best")
        ax.set_title(f"Precision-Recall curves {name}")
        plt.savefig(f'./figures/{name}')
        plt.show()

    def save(self, filename):
        """Saved trained model"""
        pickle.dump(self.model, open(filename, 'wb'))

    def load(self, path):
        self.model = pickle.load(open(path, 'rb'))
        print(f'Loaded from {path}')

    def score(self, metrics):
        """Score on test dataset"""
        y_pred = self.predict()
        return metrics(y_pred, self.test_y)

    def predict(self):
        """Prediction with optimal parameters"""
        return self.model.predict(self.test_X)

    def visualisation(self, path, metrics='mean_test_score'):
        """save figure of cv
            metrics (string) : to be chosen among https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
            https://scikit-learn.org/stable/modules/model_evaluation.html#scoring
            implement custom metrics as in https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        """

        fig = plt.figure()
        scores = [x for x in self.search.cv_results_[metrics]]
        params_values = [y for y in self.search.cv_results_['params']]
        # params_values = format_paramsdict(params_values)
        plt.xticks(np.arange(len(params_values)), params_values)
        plt.plot(scores)
        plt.xlabel('parameters')
        plt.ylabel(metrics)
        plt.title(f'{metrics} for models tested during grid search')
        plt.savefig(path)
        plt.clf()

        return None


class SVMModel(Model):
    def __init__(self, train_data, test_data, preprocess, metric='f1_micro'):
        """
        https://scikit-learn.org/stable/modules/svm.html#shrinking-svm"""
        super().__init__(train_data, test_data, preprocess, metric)
        self.model = SVC(random_state=0, probability=True)
        self.pipe = Pipeline(steps=[("preprocess", preprocess), ("model", self.model)])
        # self.pipe = Pipeline(steps=[("model", self.model)])

class RFModel(Model):
    """
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=random%20classifier#sklearn.ensemble.RandomForestClassifier"""
    def __init__(self, train_data, test_data, preprocess, metric='f1_micro'):
        super().__init__(train_data, test_data, preprocess, metric)
        self.model = RandomForestClassifier(random_state=0)
        self.pipe = Pipeline(steps=[("preprocess", preprocess), ("model", self.model)])

class LRModel(Model):
    """
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"""
    def __init__(self, train_data, test_data, preprocess, metric='f1_micro'):
        super().__init__(train_data, test_data, preprocess, metric)
        self.model = LogisticRegression(random_state=0, solver='saga')
        self.pipe = Pipeline(steps=[("preprocess", preprocess), ("model", self.model)])


class lda(Model):
    def __init__(self, train_data, test_data, preprocess, metric='f1_micro'):
        super().__init__(train_data, test_data, preprocess, metric)
        self.model = LinearDiscriminantAnalysis()
        self.pipe = Pipeline(steps=[("preprocess", preprocess), ("model", self.model)])

class qda(Model):
    def __init__(self, train_data, test_data, preprocess, metric='f1_micro'):
        super().__init__(train_data, test_data, preprocess, metric)
        self.model = QuadraticDiscriminantAnalysis()
        self.pipe = Pipeline(steps=[("preprocess", preprocess), ("model", self.model)])

class sk_NN(Model):
    def __init__(self, train_data, test_data, preprocess, metric='f1_micro'):
        """
        https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier"""
        super().__init__(train_data, test_data, preprocess, metric)
        self.model = MLPClassifier(random_state=0)
        self.pipe = Pipeline(steps=[("preprocess", preprocess), ("model", self.model)])

class LSTM(Model):
    def __init__(self, train_data, test_data, preprocess, metric='f1_micro'):
        """
        https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html"""
        
        super().__init__(train_data, test_data, preprocess, metric)
        self.model = LSTM_classifier2(random_state=0)
        self.pipe = Pipeline(steps=[("preprocess", preprocess), ("model", self.model)])

if __name__=='__main_':
    pass

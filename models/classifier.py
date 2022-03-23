from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

import matplotlib.pyplot as plt
import numpy as np
import pickle

from utils import format_paramsdict

class model():
    def __init__(self, train_data, test_data, preprocess):
        self.train_X = train_data.iloc[:, 0:64]
        self.train_y = train_data.iloc[:, 64]

        self.test_X = test_data.iloc[:, 0:64]
        self.test_y = test_data.iloc[:, 64]

        self.preprocess = preprocess
        self.parameter_optimal = {}
        self.model = None
        self.gs_result = None
    
    def grid_search(self, parameters, n_fold, scoring='accuracy'):
        """Grid search on train dataset"""
        pipe = Pipeline(steps=[("model", self.model)])
        # Parameters of pipelines can be set using ‘__’ separated parameter names:
        pipe.fit()
        search = GridSearchCV(pipe, parameters, n_jobs=-1, cv=n_fold, scoring=scoring)
        search.fit(self.train_X, self.train_y)
        
        self.model = search.best_estimator_
        self.gs_result = search

        return search.best_estimator_

    def score(self):
        """Score on test dataset"""
        pass

    def predict(self):
        """Prediction with optimal parameters"""
        pass

    def save(self, filename):
        """Saved trained model"""
        pickle.dump(self.model, open(filename, 'wb'))

    def load(self, path):
        self.model = pickle.load(open(path, 'rb'))

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
            TODO 2 visualization functions needed depending on if we use sk metrics or custom metrics that we need to recompute ? 
            Not necessarly : implement custom metrics as in https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        """

        fig = plt.figure()
        scores = [x for x in self.gs_result.cv_results_[metrics]]
        params_values = [y for y in self.gs_result.cv_results_['params']]
        # params_values = format_paramsdict(params_values)
        plt.xticks(np.arange(len(params_values)), params_values)
        plt.plot(scores)
        plt.xlabel('parameters')
        plt.ylabel(metrics)
        plt.title(f'{metrics} for models tested during grid search')
        plt.savefig(path)
        plt.clf()

        return None

class random_forest_model(model):
    def __init__(self, train_data, test_data, preprocess):
        """
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=random%20classifier#sklearn.ensemble.RandomForestClassifier"""
        super().__init__(train_data, test_data, preprocess)

        self.model = RandomForestClassifier(random_state=0)

class SVMModel(model):
    def __init__(self, train_data, test_data, preprocess):
        """
        https://scikit-learn.org/stable/modules/svm.html#shrinking-svm"""
        super().__init__(train_data, test_data, preprocess)

        self.model = SVC(random_state=0)

class logistic_regression(model):
    def __init__(self, train_data, test_data, preprocess):
        """
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"""
        super().__init__(train_data, test_data, preprocess)

        self.model = LogisticRegression(random_state=0, solver='saga')

class lda(model):
    def __init__(self, train_data, test_data, preprocess):
        super().__init__(train_data, test_data, preprocess)

        self.model = LinearDiscriminantAnalysis()

class qda(model):
    def __init__(self, train_data, test_data, preprocess):
        super().__init__(train_data, test_data, preprocess)

        self.model = QuadraticDiscriminantAnalysis()

if __name__=='__main_':
    pass

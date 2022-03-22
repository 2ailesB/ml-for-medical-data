from sklearn import preprocessing

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


import pickle

class model():
    def __init__(self, train_data, test_data, preprocess):
        self.train_X = train_data.iloc[:, 0:64]
        self.train_y = train_data.iloc[:, 64]

        self.test_X = test_data.iloc[:, 0:64]
        self.test_y = test_data.iloc[:, 64]

        self.preprocess = preprocess
        self.parameter_optimal = {}
    
    def grid_search(self, parameters):
        """Grid search on train dataset"""
        pass

    def score(self):
        """Score on test dataset"""
        pass

    def predict(self):
        """Prediction with optimal parameters"""
        pass

    def save(self):
        """Saved trained model"""
        pass

class random_forest_model(model):
    def __init__(self, train_data, test_data, preprocess, nfold):
        super().__init__(train_data, test_data, preprocess, nfold, hy)
        self.model = RandomForestClassifier(random_state=0)
        


class SVMModel(model):
    def __init__(self, train_data, test_data, preprocess):
        super().__init__(train_data, test_data, preprocess)

        self.model = SVC()

    def grid_search(self, parameters, n_flod):
        """Grid search on train dataset"""
        # model = SVC(random_state=0)
        pipe = Pipeline(steps=[("model", self.model)])

        # Parameters of pipelines can be set using ‘__’ separated parameter names:
        search = GridSearchCV(pipe, parameters, cv=n_flod, n_jobs=-1)
        search.fit(self.train_X, self.train_y)

        self.parameter_optimal = search.best_params_
        print(self.parameter_optimal)
        print(search.)
        self.model = SVC(**self.parameter_optimal)

        return search

    def score(self, metrics):
        """Score on test dataset"""
        y_pred = self.predict()
        return metrics(y_pred, self.test_y)

    def predict(self):
        """Prediction with optimal parameters"""
        self.model = SVC(self.parameter_optimal)
        return self.model(self.test_X, self.test_y)

    def save(self, filename):
        """Saved trained model"""
        pickle.dump(self.model, open(filename, 'wb'))


if __name__=='__main_':
    pass
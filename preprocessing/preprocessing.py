from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


class basic_preprocessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Numerical features to pass down the numerical pipeline
        return X

### TODO : add the parameters of the preprocessing functions from sklearn + in main 
class pca_preprocessing(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, kernel='linear'):
        super().__init__()
        self.n_components = n_components
        self.model = KernelPCA(n_components=n_components, kernel=kernel)

    def fit(self, X, y=None):
        self.model.fit(X)
        return self

    def transform(self, X, y=None):
        # Numerical features to pass down the numerical pipeline
        return self.model.transform(X)

class norm_preprocessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.model = StandardScaler()

    def fit(self, X, y=None):
        self.model.fit(X)
        return self

    def transform(self, X, y=None):
        # Numerical features to pass down the numerical pipeline
        return self.model.transform(X)

class normPCA_preprocessing(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, kernel='linear'):
        super().__init__()
        self.model = Pipeline(steps=[("scale", StandardScaler()), ("pca", KernelPCA(n_components=n_components, kernel=kernel))])

    def fit(self, X, y=None):
        self.model.fit(X)
        return self

    def transform(self, X, y=None):
        # Numerical features to pass down the numerical pipeline
        return self.model.transform(X)


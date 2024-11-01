from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


class   IndicatorCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, args):
        self.args = args
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = calc_indicators(X, self.args)
        return X


class   LabelCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, args):
        self.args = args

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = calc_labels(X, self.args)
        return X


class   Cleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.drop(X.index[:10])
        X = X.drop(X.index[-10:])
        X.reset_index(drop=True, inplace=True)
        X.bfill(inplace=True)
        return X


class   BalancedOverSampler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X

class   UnbalancedSampler(BaseEstimator, TransformerMixin):
    def __init__(self, balance_type):
        self.balance_type = balance_type

    def fit(self, X, y=None)
        return self

    def transform(self, X, y=None):
        if self.balance_type == "win":
            pass
        
        elif self.balance_type == "lose":
            pass
        return X
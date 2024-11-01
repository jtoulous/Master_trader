from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

class PreprocessingPipeline(Pipeline):
    def fit_transform(self, X, y=None, **fit_params):
        for name, transformer in self.steps:
            if transformer is not None:
                if hasattr(transformer, 'transform_y'):
                    y = transformer.transform_y(X, y)
                X = transformer.fit_transform(X, y) if y is not None else transformer.fit_transform(X)
                
        return X, y



class   IndicatorCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, args):
        self.args = args
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = calc_indicators(X, self.args)
        return X

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass


class   LabelCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, args):
        self.args = args

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = calc_labels(X, self.args)
        return X

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass


class   Cleaner(BaseEstimator, TransformerMixin):
    def __init__(self, args):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.drop(X.index[:10])
        X = X.drop(X.index[-10:])
        X.reset_index(drop=True, inplace=True)
        X.bfill(inplace=True)
        return X

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass


class   LabelExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def transform(self, X, y):
        X = X.drop(columns=['LABEL'])
        return X

    def transform_y(self, X, y):
        y = X['LABEL']
        return y

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass



#class   CustomScaler(BaseEstimator, TransformerMixin):
#    def __init__(self):
#        self.scaler = StandardScaler()
#
#    def fit(self, X, y):
#        self.scaler.fit(X)
#    
#    def transform(self, X, y=None):
#        X = self.scaler.transform(X)
#        return X
#
#    def __getstate__(self):
#        return self.scaler.__getstate__()
#
#    def __setstate__(self, state):
#        self.scaler.__setstate__(state)


class   CustomSampler(BaseEstimator, TransformerMixin):
    def __init__(self, sampler_type):
        if sampler_type == "oversampler":
            self.sampler = RandomOverSampler(sampling_strategy='auto')

    def fit(self, X, y):
        self.sampler.fit(X)
    
    def transform(self, X, y):
        X = self.sampler.transform(X)
        return X, y

    def __getstate__(self):
        return self.sampler.__getstate__()

    def __setstate__(self, state):
        self.sampler.__setstate__(state)


training_pipeline_balanced = Pipeline([
    ('indicators_calculator', IndicatorCalculator(args)),
    ('label_calculator', LabelCalculator(args)),
    ('cleaner', Cleaner()),
    ('label_extractor', LabelExtractor()),
    ('over_sampler', CustomSampler('oversampler')),
])
    
    ('scaler', StandardScaler()),
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np

class BoostedHybrid(BaseEstimator):
    def __init__(self, model_1, model_2, features_1, features_2, sku = 'sku'):
        self.model_1 = model_1
        self.model_2 = model_2
        self.features_1 = features_1
        self.features_2 = features_2
        self.grouper = None
        self.sku = sku
        self.target = None
        self.y_columns = None # store column names from fit method
        
    def fit(self, X, y):
        # fit 1 model
        self.grouper = y.index.name
        self.target = y.name
        
        y = pd.concat([X, y], axis = 1)
        y = y.groupby([self.grouper, self.sku])[self.target].mean().unstack(self.sku)
        
        X_1 = y.join(X).loc[:, self.features_1].drop_duplicates()
        X_2 = X.loc[:, self.features_2]
        self.model_1.fit(X_1, y)

        y_fit = pd.DataFrame(
        self.model_1.predict(X_1),
        index=X_1.index, columns=y.columns,
        )

        # residuals
        y_resid = y - y_fit
        y_resid = y_resid.stack().squeeze()

        # fit self.model_2 on residuals
        self.model_2.fit(X_2, y_resid)

        # Save data
        self.y_columns = y.columns
        self.y_fit = y_fit
        self.y_resid = y_resid

    def predict(self, X):
        X_1 = X.groupby(self.grouper).mean().loc[:, self.features_1]
        X_2 = X.loc[:, self.features_2]
        y_pred = pd.DataFrame(
            self.model_1.predict(X_1),
            index=X_1.index, columns=self.y_columns)
        y_pred = y_pred.stack().squeeze()  # wide to long
        y_pred += self.model_2.predict(X_2)
    
        return y_pred.clip(0.0)
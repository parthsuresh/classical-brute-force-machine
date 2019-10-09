from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit, GridSearchCV

class LinearRegressionModel():
    def __init__(self, X_train, y_train):
        self.lr_model = LinearRegression().fit(X_train, y_train)

    def get_params(self):
        return self.lr_model.get_params()

    def predict(self, X_test):
        return self.lr_model.predict(X_test)

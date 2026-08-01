import numpy as np

class WeightedVotingRegressor:
    """
    Custom weighted voting regressor to allow grid-searched weight optimization
    and easy serialization.
    """
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights
        if weights is not None:
            self.weights = np.array(weights) / sum(weights)

    def fit(self, X, y):
        # Base models are assumed to be pre-trained
        pass

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        return np.average(predictions, axis=0, weights=self.weights)

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class NaiveBayesClassifier:
    """
    Class to represent a Naive Bayes Classifier.
    """

    def __init__(self, train_features: pd.DataFrame, train_labels: pd.Series) -> None:
        self.train_features = train_features
        self.train_labels = train_labels
        self.train_dataset = pd.concat([self.train_features, self.train_labels], axis=1)

    @property
    def n_train_samples(self) -> int:
        return len(self.train_features.index)

    @property
    def features(self) -> pd.Index:
        return self.train_features.columns

    @property
    def labels(self) -> np.ndarray:
        return self.train_labels.unique()

    @property
    def prior(self) -> pd.DataFrame:
        """
        Method to calculate the prior probability associated to each label.

        Returns
        -------
        priors: pd.DataFrame
            A dataframe where each label is mapped to its prior.
        """
        _prior = pd.DataFrame(columns=self.labels.tolist())
        for label in self.labels:
            _prior[label] = len(self.train_labels[self.train_labels == label]) / len(
                self.train_labels
            )
        return _prior

    def likelihood(self):
        def gaussian_pdf(x: pd.Series, mu: float, sigma_squared: float) -> pd.Series:
            p: pd.Series = (1 / np.sqrt(2 * np.pi * sigma_squared)) * np.exp(
                -((x - mu) ** 2) / (2 * sigma_squared)
            )
            p.name = "gaussian probability distribution"
            return p

        _likelihood = dict((label, 1) for label in self.labels)
        _mean = pd.DataFrame(columns=self.labels.tolist())
        _variance = pd.DataFrame(columns=self.labels.tolist())
        for label in self.labels:
            _features_per_label = self.train_features[label == self.train_labels]
            _mean[label] = _features_per_label.mean()
            _variance[label] = _features_per_label.var()
            for feature in self.features:
                _likelihood[label] *= gaussian_pdf(
                    _features_per_label[feature],
                    _mean[label][feature],
                    _variance[label][feature],
                )

        return _likelihood

    @property
    def normalization(self):
        _norm = 0
        for label in self.labels:
            _norm += self.likelihood()[label] * self.prior[label]
        return _norm

    @property
    def posterior(self) -> pd.DataFrame:
        """
        Method to calculate the posterior probability associated to each label.

        Returns
        -------
        posteriors: pd.DataFrame
            A series where each label is mapped to its posterior.
        """
        _posterior = pd.DataFrame(columns=self.labels.tolist())
        for label in self.labels:
            _posterior[label] = (
                self.prior[label] * self.likelihood() / self.normalization
            )

    def predict(
        self,
        test_features: pd.DataFrame,
    ):
        pass

    def print_accuracy(self, test_labels: pd.Series):
        pass


if __name__ == "__main__":
    cancer = load_breast_cancer()
    all_features = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
    all_target = pd.Series(cancer.target, name="diagnosis")
    dataset = pd.concat([all_features.iloc[:, 0:4], all_target], axis=1)
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=41)
    train_features = train_data.iloc[:, :-1]
    train_target = train_data.iloc[:, -1]
    test_features = test_data.iloc[:, :-1]
    test_target = test_data.iloc[:, -1]

    nbc = NaiveBayesClassifier(train_features=train_features, train_labels=train_target)
    print("Prior:")
    print(nbc.prior)
    print("Likelihood:")
    print(nbc.likelihood())
    print("Normalization:")
    print(nbc.normalization)
    print("Posterior:")
    print(nbc.posterior)

import numpy as np
import pandas as pd
import scipy

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

import matplotlib.pyplot as plt


class NaiveBayesClassifier:
    """
    Class to represent a Naive Bayes Classifier.
    """

    def __init__(self, dataset: pd.DataFrame, test_size: float = 0.2) -> None:
        """
        Initialization of the Naive Bayes Classifier instance.
        Input validation and train-test split.

        Parameters
        ----------
        dataset : pd.DataFrame

        test_size: float
            default = 0.2
            must be between 0.0 and 1.0
        """
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError("The dataset must be a pandas DataFrame.")
        if not isinstance(test_size, float):
            raise ValueError("The test_size must be a floating number")
        if test_size < 0.0:
            raise ValueError("The test_size must be greater than 0.")
        if test_size > 1.0:
            raise ValueError("The test_size must be less than 1.")
        train_data, test_data = train_test_split(
            dataset, test_size=test_size, random_state=41
        )
        self.train_features: pd.DataFrame = train_data.iloc[:, :-1]
        self.train_labels: pd.Series = train_data.iloc[:, -1]
        self.test_features: pd.DataFrame = test_data.iloc[:, :-1]
        self.test_labels: pd.Series = test_data.iloc[:, -1]

    def plot_features_histograms(self) -> None:
        def draw_histograms(df, variables, n_rows, n_cols):
            fig = plt.figure()
            for i, var_name in enumerate(variables):
                ax = fig.add_subplot(n_rows, n_cols, i + 1)
                df[var_name].hist(bins=10, ax=ax)
                ax.set_title("Distribution of " + var_name)
            fig.tight_layout()  # Improves appearance a bit.
            plt.show()

        draw_histograms(self.train_features, self.features, 2, 2)

    @property
    def n_train_samples(self) -> int:
        return len(self.train_features.index)

    @property
    def n_test_samples(self) -> int:
        return len(self.test_features.index)

    @property
    def features(self) -> pd.Index:
        return self.train_features.columns

    @property
    def labels(self) -> np.ndarray:
        return sorted(self.train_labels.unique())

    @property
    def prior(self) -> pd.DataFrame:
        """
        Method to calculate the prior probability associated to each label.
        It takes into consideration only the TRAIN labels.

        Returns
        -------
        priors: pd.DataFrame
            A dataframe where each label is mapped to its prior.
        """
        _prior = pd.DataFrame(columns=self.labels)
        for label in self.labels:
            _prior[label] = pd.Series(
                len(self.train_labels[self.train_labels == label])
                / len(self.train_labels)
            )
        return _prior

    def likelihood(self) -> pd.DataFrame:
        """
        Method to calculate the likelihood.

        Returns
        -------
        A pandas DataFrame with a probability for each label and each test observation.
        """

        # def gaussian_pdf(x: float, mu: float, sigma_squared: float) -> pd.Series:
        #     sigma = np.sqrt(sigma_squared)
        #     numerator = np.exp(-(((x - mu) / sigma) ** 2) * 0.5)
        #     denominator = np.sqrt(2 * np.pi * sigma_squared)
        #     pdf = numerator / denominator
        #     return pdf

        _likelihood = pd.DataFrame(
            data=np.ones(shape=(len(self.test_features.index), len(self.labels))),
            columns=self.labels,
        )
        _mean = pd.DataFrame(columns=self.labels)
        _variance = pd.DataFrame(columns=self.labels)
        for label in self.labels:
            _features_per_label = self.train_features[label == self.train_labels]
            _mean[label] = _features_per_label.mean()
            _variance[label] = _features_per_label.var()
            for feature in self.features:
                # _likelihood[label] *= gaussian_pdf(
                #     self.test_features[feature],
                #     _mean[label][feature],
                #     _variance[label][feature],
                # )
                _likelihood[label] *= scipy.stats.norm(
                    _mean[label][feature], _variance[label][feature]
                ).pdf(self.test_features[feature])
        assert not _likelihood.empty
        return _likelihood

    @property
    def normalization(self) -> pd.Series:
        _norm = 0
        for label in self.labels:
            _norm += self.likelihood()[label] * self.prior[label].values
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
        _posterior = pd.DataFrame(columns=self.labels)
        for label in self.labels:
            _posterior[label] = (
                self.prior[label].values * self.likelihood()[label] / self.normalization
            )
        return _posterior

    def prediction(self) -> pd.Series:
        pred = self.posterior.apply(np.argmax, axis=1)
        return pred

    def print_accuracy(self) -> None:
        my_accuracy_score = accuracy_score(
            self.test_labels, self.prediction(), normalize=True
        )
        my_confusion_matrix = confusion_matrix(self.test_labels, self.prediction())
        my_f1_score = f1_score(self.test_labels, self.prediction())

        print("{0:<15} {1:>15}".format("my accuracy", my_accuracy_score))
        print("{0:<15} {1:>15}".format("my f1", my_f1_score))
        print("my confusion matrix: \n", my_confusion_matrix)


if __name__ == "__main__":
    cancer = load_breast_cancer()
    all_features = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
    all_target = pd.Series(cancer.target, name="diagnosis")
    dataset = pd.concat([all_features.iloc[:, 0:4], all_target], axis=1)

    bayes = NaiveBayesClassifier(dataset=dataset)
    bayes.plot_features_histograms()
    print("Prior:")
    print(bayes.prior)
    print("-------------------------------------")
    print("Likelihood:")
    print(bayes.likelihood())
    print("-------------------------------------")
    print("Normalization:")
    print(bayes.normalization)
    print("-------------------------------------")
    print("Posterior:")
    print(bayes.posterior)
    print("-------------------------------------")
    print("Prediction:")
    print(bayes.prediction())
    bayes.print_accuracy()

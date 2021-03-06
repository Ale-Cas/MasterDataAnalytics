import numpy as np
import pandas as pd

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
        self.train_data, self.test_data = train_test_split(
            dataset, test_size=test_size, random_state=41
        )
        self.train_features: pd.DataFrame = self.train_data.iloc[:, :-1]
        self.train_labels: pd.Series = self.train_data.iloc[:, -1]
        self.test_features: pd.DataFrame = self.test_data.iloc[:, :-1]
        self.test_labels: pd.Series = self.test_data.iloc[:, -1]

    def plot_features_histograms(self) -> None:
        def draw_histograms(df, variables, n_rows: int = 1, n_cols: int = 1):
            fig = plt.figure()
            for i, var_name in enumerate(variables):
                ax = fig.add_subplot(n_rows, n_cols, i + 1)
                df[var_name].hist(bins=10, ax=ax)
                ax.set_title("Distribution of " + var_name)
            fig.tight_layout()  # Improves appearance a bit.
            plt.show()

        draw_histograms(self.train_features, self.features, len(self.features), 1)

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

        def gaussian_pdf(x: float, mu: float, sigma: float) -> pd.Series:
            numerator = np.exp(-(((x - mu) / sigma) ** 2) * 0.5)
            denominator = np.sqrt(2 * np.pi) * sigma
            pdf = numerator / denominator
            return pdf

        _likelihood = pd.DataFrame(
            data=np.ones(shape=(len(self.test_features.index), len(self.labels))),
            columns=self.labels,
        )
        _mean = pd.DataFrame(columns=self.labels)
        _std_dev = pd.DataFrame(columns=self.labels)
        for label in self.labels:
            _features_per_label = self.train_features[label == self.train_labels]
            _mean[label] = _features_per_label.mean()
            _std_dev[label] = _features_per_label.std()
            for feature in self.features:
                for idx in range(len(self.test_features)):
                    _likelihood.iloc[idx][label] *= gaussian_pdf(
                        x=self.test_features.iloc[idx][feature],
                        mu=_mean[label][feature],
                        sigma=_std_dev[label][feature],
                    )
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
    diagnosis = pd.Series(cancer.target, name="diagnosis")
    sel_features = ["mean radius", "mean texture", "mean smoothness"]
    dataset = pd.concat([all_features.loc[:, sel_features], diagnosis], axis=1)

    bayes = NaiveBayesClassifier(dataset=dataset)
    print("Data:")
    print(dataset)
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

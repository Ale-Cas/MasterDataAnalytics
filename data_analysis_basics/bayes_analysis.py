import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class BayesAnalysis:
    def __init__(self, features: pd.DataFrame, target: pd.Series) -> None:
        self.features = features
        self.target = target

    @property
    def prior(self) -> float:
        # when diagnosis is negative
        return len(self.target[self.target == 1]) / len(self.target)

    @property
    def complementary_prior(self) -> float:
        # when diagnosis is negative
        return len(self.target[self.target == 0]) / len(self.target)

    def likelihood(self) -> None:
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
    bayes = BayesAnalysis(features=train_features, target=train_target)
    assert (
        bayes.prior + bayes.complementary_prior == 1.0
    ), "The sum of these two probabilities must be 1, because they are complementary"
    print(bayes.prior)

import sklearn.datasets
import sklearn.metrics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CLASSES = [1, 2]
COLORS = plt.get_cmap("tab10").colors


def load_iris_dataset():
    data = sklearn.datasets.load_iris()
    X = pd.DataFrame(data=data.data, columns=data.feature_names)

    # Simplify feature names; removes " (cm)" suffix
    X = X.rename({x: x[:-5].replace(" ", "_") for x in data.feature_names}, axis=1)
    y = pd.Series(data.target, name="y")
    y = y.loc[y.isin(CLASSES)]
    y = y.replace({x: i for i, x in enumerate(CLASSES)})
    X = X.loc[y.index]
    return X.reset_index(drop=True), y.reset_index(drop=True)


def scatter_iris(X, y, features):
    assert len(features) == 2, f"Two features are required! You gave me {features}."
    assert set(features).issubset(X.columns), "You gave me incorrect feature names! Try again."
    plt.figure()
    for value, color in zip(sorted(y.unique()), COLORS):
        view = X.loc[y == value]
        label = f"$y = {value}$"
        plt.scatter(x=view[features[0]], y=view[features[1]], color=color, label=label)
    plt.xlabel(f"{features[0]} (cm)")
    plt.ylabel(f"{features[1]} (cm)")
    plt.legend()


def linear_separate_iris(X, y, features, decision_boundary_fct):
    scatter_iris(X, y, features)
    x_min = X[features[0]].min()
    x_max = X[features[0]].max()
    x = np.linspace(x_min, x_max)
    y = decision_boundary_fct(x)
    plt.plot(x, y, color="black", label="decision boundary")
    plt.legend()


def main():
    X, y = load_iris_dataset()
    features = ["sepal_length", "petal_width"]
    scatter_iris(X, y, features)

    def linear_function(x):
        m = -0.1
        b = 2.3
        return m * x + b

    def predict_linear_function(X):
        return (linear_function(X.iloc[:, 0]) < X.iloc[:, 1]).astype(int)
    linear_separate_iris(X, y, features, linear_function)
    y_pred_db = predict_linear_function(X[features])
    precision_db = sklearn.metrics.precision_score(y, y_pred_db)
    recall_db = sklearn.metrics.recall_score(y, y_pred_db)
    print(precision_db, recall_db)
    plt.show()


if __name__ == "__main__":
    main()

import sklearn.linear_model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLORS = plt.get_cmap("tab10").colors
XMAX = 10


def load_data(n=200, m=3, b=1, noise=2):
    rng = np.random.default_rng()
    x = XMAX * rng.random(n)
    y = m * x + b + noise * rng.standard_normal(n)
    X = pd.DataFrame(x, columns=["x_1"])
    y = pd.Series(y, index=X.index, name="y")
    return X, y


def scatter_data(X, y):
    X = np.array(X)
    plt.figure()
    plt.grid()
    plt.scatter(x=X[:, 0], y=y, color=COLORS[0], label="data")
    plt.xlabel("$x_1$")
    plt.ylabel("$y$")
    plt.legend()


def plot_prediction(X, y, reg):
    scatter_data(X, y)
    x = np.linspace(0, XMAX)
    if type(reg) == sklearn.linear_model.LinearRegression:
        label = "LR model"
        y = reg.predict(pd.DataFrame(x, columns=["x_1"]))
    else:
        label = "our model"
        y = reg(x)
    plt.plot(x, y, color=COLORS[1], label=label)
    plt.xlabel("$x_1$")
    plt.ylabel("$y$")
    plt.legend()


def plot_prediction_error(X, y, fn):
    X_frac = X.sample(n=10)
    y_frac = y.loc[X_frac.index]
    y_pred = fn(X_frac).squeeze()

    ymin = np.minimum(y_frac, y_pred)
    ymax = np.maximum(y_frac, y_pred)
    x = np.array(X_frac)[:, 0]
    plt.scatter(x, y_frac, label="data")
    plt.plot(x, y_pred, c=COLORS[1], label="our fitted model")
    plt.vlines(x, ymin, ymax, color=COLORS[2], label="prediction error")
    plt.legend()

import sklearn.datasets
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.pipeline

import numpy as np
import matplotlib.pyplot as plt

COLORS = plt.get_cmap("tab10").colors
LIMIT = 1


def load_nonlinear_cos_dataset(n=30, noise=0.1, seed=42, use_cos=False):
    rng = np.random.default_rng(seed=seed)
    x = rng.uniform(low=-LIMIT, high=LIMIT, size=n)
    x = np.sort(x)
    y = np.cos(2 * np.pi * x / (LIMIT))
    return x.reshape(n, 1), y + np.random.normal(0, noise, n)


def load_nonlinear_pol_dataset(n=30, k=3, noise=0.05, seed=42):
    rng = np.random.default_rng(seed=seed)
    x = rng.uniform(low=-LIMIT, high=LIMIT, size=n)
    x = np.sort(x)

    roots = np.linspace(-LIMIT, LIMIT, k)
    terms = [x - root for root in roots]
    y = np.prod(terms, axis=0)
    return x.reshape(n, 1), y + np.random.normal(0, noise, n)


def scatter_data(X, y, ax=None):
    if ax is None:
        plt.figure()
        ax = plt.gca()
    ax.grid()
    ax.scatter(x=X[:, 0], y=y, color=COLORS[0], label="data")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$y$")
    ax.set_ylim([-1.5, 1.5])
    ax.legend()


def plot_prediction(X, y, est):
    scatter_data(X, y)
    x = np.linspace(-LIMIT, LIMIT, 1000)
    y_pred = est.predict(x.reshape(1000, 1))
    plt.plot(x, y_pred, color=COLORS[1], label="model")
    plt.ylim([-1.5, 1.5])
    plt.legend()

    # y_pred = est.predict(X)
    # plt.plot(X[:, 0], y_pred, color=COLORS[1], label=r"$\hat{y}$")


def main():
    n = 30
    k = 3
    X, y = load_nonlinear_pol_dataset(n, k=k)
    # scatter_data(X, y)
    scatter_data(*load_nonlinear_pol_dataset(n=1000, seed=1))
    scatter_data(*load_nonlinear_pol_dataset(n=1000, seed=2))
    scatter_data(*load_nonlinear_pol_dataset(n=1000, seed=3))

    poly = sklearn.preprocessing.PolynomialFeatures(degree=k)
    lr = sklearn.linear_model.LinearRegression()
    pipe = sklearn.pipeline.make_pipeline(poly, lr)
    pipe.fit(X, y)
    print(lr.coef_)

    plot_prediction(X, y, pipe)
    plt.show()


if __name__ == "__main__":
    main()

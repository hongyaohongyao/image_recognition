import numpy as np
from sklearn.datasets import make_moons
from platt_svm import SVM
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)


def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)


if __name__ == "__main__":

    """
    start calc 2023-01-03 15:59:28.469280
    end calc 2023-01-03 15:59:30.063595
    """

    X, y = make_moons(n_samples=1000, noise=0.15, random_state=42)

    # model = SVM(kernel='poly', C=5, max_iter=500, coef0=1)
    model = SVM(kernel='rbf', C=5, max_iter=40, gamma='scale')
    # model = SVC(kernel='rbf', C=5)
    model.fit(X, y)

    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    plot_predictions(model, [-1.5, 2.5, -1, 1.5])

    plt.savefig("svm_nolinear.png")

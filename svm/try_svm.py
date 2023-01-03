import numpy as np

from svm import SVM
import matplotlib.pyplot as plt


def plot_svc_decision_boundary(model, xmin, xmax, sv=True):
    w = model.w
    b = model.b

    x0 = np.linspace(xmin, xmax, 200)


if __name__ == "__main__":

    np.random.seed(12)
    num_observations = 50

    x1 = np.random.multivariate_normal([1, 1], [[1, .75], [.75, 1]],
                                       num_observations)
    x2 = np.random.multivariate_normal([9, 9], [[1, .75], [.75, 1]],
                                       num_observations)

    X = np.vstack((x1, x2)).astype(np.float32)
    X = (X - X.mean()) / X.std()
    y = np.hstack((np.zeros(num_observations), np.ones(num_observations)))

    model = SVM(kernel='linear', C=1, max_iter=500)
    model.fit(X, y)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[:, 0], X[:, 1], c=y, alpha=.4)

    w, b, sv = model.w[0], model.b[0], model.support_vec[0]
    x1, _ = max(X, key=lambda x: x[0])
    x2, _ = min(X, key=lambda x: x[0])
    a1, a2 = w
    margin = 1 / a2
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    ax.plot([x1, x2], [y1, y2], 'k-')
    ax.plot([x1, x2], [y1 - margin, y2 - margin], 'k--')
    ax.plot([x1, x2], [y1 + margin, y2 + margin], 'k--')

    for i, s in enumerate(sv):
        x, y = s
        ax.scatter([x], [y],
                   s=150,
                   c='none',
                   alpha=0.7,
                   linewidth=1.5,
                   edgecolor='#AB3319')

    plt.savefig("svm.png")

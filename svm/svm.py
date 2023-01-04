import numpy as np
import random
from datetime import datetime
from numba import jit


@jit(nopython=True)
def clip(alpha, L, H):
    if alpha < L:
        return L
    elif alpha > H:
        return H
    else:
        return alpha


@jit(nopython=True)
def g(i, gram_K, alphas, y, b):
    "SVM classifier y = w^Tx + b"
    # Kernel function vector.
    ks = gram_K[:, i]  #N1,N2 -> N1

    # Predictive value. w
    wx = np.dot(alphas * y, ks.T)
    gx = wx + b
    return gx


class SVM(object):

    def __init__(self,
                 C=1,
                 max_iter=50,
                 kernel='rbf',
                 degree=3,
                 gamma='scale',
                 coef0=0):
        super(SVM, self).__init__()
        self.C = C
        self.max_iter = max_iter
        self.alphas = []
        self.support_vec = []
        self.w = []
        self.b = []
        self.gamma = gamma
        self.degree = degree
        self._gamma = 0
        self.coef0 = coef0
        self.class_num = 0
        self.linear_kernel = kernel == 'linear'
        self.kernel = kernel

    def append_result(self, alphas, X, y, b):
        self.alphas.append(alphas)
        self.support_vec.append(X)
        if self.linear_kernel:
            self.w.append(np.dot(y * alphas, X))
        else:
            self.w.append(y * alphas)
        self.b.append(b)

    def fit(self, X, y, verbose=False):
        assert len(self.alphas) == 0, "SVM Model has been fitted"
        X, y = np.array(X, dtype=float), np.array(y, dtype=int)

        self.class_num = np.max(y) + 1  # number of class
        assert self.class_num >= 2, 'class number should be larger than 2'
        # create kernel
        gamma = self.gamma
        if gamma == "scale":
            gamma = 1 / (X.shape[1] * X.var())
        elif gamma == 'auto':
            gamma = 1 / X.shape[1]
        self._gamma = gamma

        # print(f'kernel start calc {datetime.now()}')
        gram_K = self._kernel_matrix(X)  # numba speed 6.46 => 1.54
        # print(f'kernel end calc {datetime.now()}')

        if self.class_num == 2:
            y = np.where(y <= 0, -1, 1)  # transform label to -1 and 1
            # print(f'start calc {datetime.now()}')
            # numba speed 60.12 => 24.76
            alphas, X, y, b = self._fit(X, y, gram_K, self.max_iter, self.C)
            # print(f'end calc {datetime.now()}')
            self.append_result(alphas, X, y, b)
        else:
            """
            multi class
            """
            y = np.eye(self.class_num)[y]
            y = np.where(y <= 0, -1, 1)  # transform label to -1 and 1
            for i in range(self.class_num):
                print(f'start calc {datetime.now()}')
                alphas, X_, y_, b = self._fit(X, y[:, i], gram_K,
                                              self.max_iter, self.C)
                print(f'end calc {datetime.now()}')
                self.append_result(alphas, X_, y_, b)
                if verbose:
                    print(f"trained class{i}")

        return self

    @staticmethod
    @jit(nopython=True)
    def _fit(X, y, gram_K, max_iter, C):
        # init param
        N, _ = X.shape
        alphas = np.zeros(N)
        b = 0
        it = 0
        while it < max_iter:
            pair_changed = 0
            for i in range(N):
                a_i, y_i = alphas[i], y[i]
                gx_i = g(i, gram_K, alphas, y, b)
                E_i = gx_i - y_i
                j = i
                while j == i:
                    j = int(random.uniform(0, N))
                a_j, y_j = alphas[j], y[j]

                gx_j = g(j, gram_K, alphas, y, b)
                E_j = gx_j - y_j

                K_ii, K_jj, K_ij = gram_K[i, i], gram_K[j, j], gram_K[i, j]
                eta = K_ii + K_jj - 2 * K_ij
                if eta <= 0:
                    continue
                # calculate new alpha j
                a_i_old, a_j_old = a_i, a_j
                a_j_new = a_j_old + y_j * (E_i - E_j) / eta

                # clip alpha
                if y_i != y_j:
                    L = max(0, a_j_old - a_i_old)
                    H = min(C, C + a_j_old - a_i_old)
                else:
                    L = max(0, a_i_old + a_j_old - C)
                    H = min(C, a_j_old + a_i_old)

                a_j_new = clip(a_j_new, L, H)
                a_i_new = a_i_old + y_i * y_j * (a_j_old - a_j_new)

                if abs(a_j_new - a_j_old) < 0.00001:
                    continue

                alphas[i], alphas[j] = a_i_new, a_j_new

                # update threshold b
                b_i = -E_i - y_i * K_ii * (a_i_new - a_i_old) - y_j * K_ij * (
                    a_j_new - a_j_old) + b
                b_j = -E_j - y_i * K_ij * (a_i_new - a_i_old) - y_j * K_jj * (
                    a_j_new - a_j_old) + b

                if 0 < a_i_new < C:
                    b = b_i
                elif 0 < a_j_new < C:
                    b = b_j
                else:
                    b = (b_i + b_j) / 2

                pair_changed += 1

            if pair_changed == 0:
                it += 1
            else:
                it = 0

        support_vec_idx = alphas > 1e-3
        alphas = alphas[support_vec_idx]
        X = X[support_vec_idx]
        y = y[support_vec_idx]

        return alphas, X, y, b

    def _kernel_matrix(self, X):
        if self.kernel == 'linear':
            return self.linear_matrix(X)
        elif self.kernel == 'rbf':
            return self.rbf_matrix(X, self._gamma)
        elif self.kernel == 'poly':
            return self.poly_matrix(X, self.degree, self.coef0)
        else:
            raise NotImplementedError

    @staticmethod
    @jit(nopython=True)
    def linear_matrix(X):
        N, _ = X.shape
        K = np.zeros((N, N))
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = np.dot(x_i, x_j.T)
        return K

    @staticmethod
    @jit(nopython=True)
    def rbf_matrix(X, gamma):
        N, _ = X.shape
        K = np.zeros((N, N))
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i,
                  j] = np.exp(-gamma * np.linalg.norm(np.subtract(x_i, x_j)))
        return K

    @staticmethod
    @jit(nopython=True)
    def rbf_array(support_vec, X, gamma):
        Nsv, _ = support_vec.shape
        N, _ = X.shape
        K = np.zeros((Nsv, N))
        for i_sv, sv in enumerate(support_vec):
            for j, x in enumerate(X):
                K[i_sv,
                  j] = np.exp(-gamma * np.linalg.norm(np.subtract(sv, x)))
        return K

    @staticmethod
    @jit(nopython=True)
    def poly_matrix(X, degree, coef0):
        N, _ = X.shape
        K = np.zeros((N, N))
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = (coef0 + np.dot(x_i, x_j.T))**degree
        return K

    @staticmethod
    @jit(nopython=True)
    def poly_array(support_vec, X, degree, coef0):
        Nsv, _ = support_vec.shape
        N, _ = X.shape
        K = np.zeros((Nsv, N))
        for i_sv, sv in enumerate(support_vec):
            for j, x in enumerate(X):
                K[i_sv, j] = (coef0 + np.dot(sv, x.T))**degree

        return K

    def _kernel_array(self, support_vec, X):
        if self.kernel == 'rbf':
            return self.rbf_array(support_vec, X, self._gamma)
        elif self.kernel == 'poly':
            return self.poly_array(support_vec, X, self.degree, self.coef0)
        else:
            raise NotImplementedError

    def predict(self, X):
        result = []
        N, _ = X.shape
        for i in range(len(self.alphas)):
            w = self.w[i]
            b = self.b[i]
            if self.linear_kernel:
                result.append(np.inner(w, X) + b)
            else:
                support_vec = self.support_vec[i]
                K = self._kernel_array(support_vec, X)
                result.append(np.dot(w, K) + b)
        if len(result) == 1:
            return np.where(result[0] <= 0, 0, 1)
        else:
            return np.argmax(np.stack(result).T, axis=-1)


if '__main__' == __name__:
    svm = SVM()
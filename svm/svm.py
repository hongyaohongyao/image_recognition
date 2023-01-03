import kernel as _KERNEL

import numpy as np
import random

class SVM(object):

    def __init__(self, C=1, max_iter=40, kernel='rbf', **kwargs):
        super(SVM, self).__init__()
        self.C = C
        self.max_iter = max_iter
        self.alphas = []
        self.support_vec = []
        self.w = []
        self.b = []
        self.class_num = 0
        self.linear_kernel = kernel == 'linear'
        self.kernel_args = kwargs
        self.kernel = kernel


    def fit(self, X, y):
        assert len(self.alphas) == 0, "SVM Model has been fitted"
        X, y = np.array(X, dtype=float), np.array(y, dtype=int)

        self.class_num = np.max(y) + 1  # 最大的类
        assert self.class_num >= 2, 'class number should be larger than 2'
        # create kernel
        if "gamma" in self.kernel_args:
            gamma = self.kernel_args['gamma']
            if gamma == "scale":
                gamma = 1 / (X.shape[1] * X.var())
            elif gamma == 'auto':
                gamma = 1 / X.shape[1]
            self.kernel_args['gamma'] = gamma

        self.kernel = getattr(_KERNEL, self.kernel)(**self.kernel_args)
        gram_K = self._kernel_matrix(X)

        if self.class_num == 2:
            y = np.where(y <= 0, -1, 1)  # 标签转换为-1和1
            self._fit(X, y, gram_K)
        else:
            """
            multi class
            """
            y = np.eye(self.class_num)[y]
            y = np.where(y <= 0, -1, 1)  # 标签转换为-1和1
            for i in range(self.class_num):
                self._fit(X, y[:, i], gram_K)

        return self

    @staticmethod
    def clip(alpha, L, H):
        if alpha < L:
            return L
        elif alpha > H:
            return H
        else:
            return alpha

    def _fit(self, X, y, gram_K):
        # 初始化参数

        N, _ = X.shape
        alphas = np.zeros(N)
        b = 0
        it = 0
        C = self.C

        l = list(range(N))

        def g(i):
            "SVM分类器函数 y = w^Tx + b"
            # Kernel function vector.
            ks = gram_K[:, i]  #N1,N2 -> N1

            # Predictive value. w
            wx = np.inner(alphas * y, ks)
            gx = wx + b
            return gx

        # all_alphas, all_bs = [], []

        while it < self.max_iter:
            pair_changed = 0
            for i in range(N):
                a_i, y_i = alphas[i], y[i]
                gx_i = g(i)
                E_i = gx_i - y_i

                j = random.choice(l[:i] + l[i + 1:])
                a_j, y_j = alphas[j], y[j]

                gx_j = g(j)
                E_j = gx_j - y_j

                K_ii, K_jj, K_ij = gram_K[i, i], gram_K[j, j], gram_K[i, j]
                eta = K_ii + K_jj - 2 * K_ij
                if eta <= 0:
                    print('WARNING  eta <= 0')
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

                a_j_new = self.clip(a_j_new, L, H)
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

                # all_alphas.append(alphas)
                # all_bs.append(b)

                pair_changed += 1
                # print('INFO   iteration:{}  i:{}  pair_changed:{}'.format(
                #     it, i, pair_changed))

            if pair_changed == 0:
                it += 1
            else:
                it = 0
            # print('iteration number: {}'.format(it))

        support_vec_idx = alphas > 1e-3
        alphas = alphas[support_vec_idx]
        X = X[support_vec_idx]
        y = y[support_vec_idx]

        self.alphas.append(alphas)
        self.support_vec.append(X)
        if self.linear_kernel:
            self.w.append(np.dot(y * alphas, X))
        else:
            self.w.append(y * alphas)
        self.b.append(b)

    def _kernel_matrix(self, X):
        N, _ = X.shape
        K = np.zeros((N, N))
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = self.kernel(x_i, x_j)
        return K

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
                Nsv, _ = support_vec.shape
                K = np.zeros((Nsv, N))
                for i_sv, sv in enumerate(support_vec):
                    for j, x in enumerate(X):
                        K[i_sv, j] = self.kernel(sv, x)
                result.append(np.dot(w, K) + b)
        if len(result) == 1:
            return np.where(result[0] <= 0, 0, 1)
        else:
            return np.argmax(np.stack(result).T, axis=-1)


if '__main__' == __name__:
    svm = SVM()
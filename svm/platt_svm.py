import numpy as np
import random
from datetime import datetime
from tqdm import tqdm
import numba
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
    "SVM分类器函数 y = w^Tx + b"
    # Kernel function vector.
    ks = gram_K[:, i]  #N1,N2 -> N1

    # Predictive value. w
    wx = np.dot(alphas * y, ks.T)
    gx = wx + b
    return gx


@jit(nopython=True)
def select_j(i, N, C, alphas, errors):
    ''' 通过最大化步长的方式来获取第二个alpha值的索引.
        '''
    valid_indices = [i for i, a in enumerate(alphas) if 0 < a < C]

    if len(valid_indices) > 1:
        j = -1
        max_delta = 0
        for k in valid_indices:
            if k == i:
                continue
            delta = abs(errors[i] - errors[j])
            if delta > max_delta:
                j = k
                max_delta = delta
    else:
        j = i
        while j == i:
            j = int(random.uniform(0, N))
    return j


@jit(nopython=True)
def get_error(i, gram_K, y, alphas, b):
    ''' 
    获取第i个数据对应的误差.
    '''
    return g(i, gram_K, alphas, y, b) - y[i]


@jit(nopython=True)
def examine_example(i, alphas, gram_K, errors, X, y, tolerance, C, N, b):
    ''' 给定第一个alpha，检测对应alpha是否符合KKT条件并选取第二个alpha进行迭代.
        '''
    E_i, y_i, alpha = errors[i], y[i], alphas[i]
    r = E_i * y_i

    # 是否违反KKT条件
    if (r < -tolerance and alpha < C) or (r > tolerance and alpha > 0):
        j = select_j(i, N, C, alphas, errors)
        ''' 对选定的一对alpha对进行优化.
            '''
        for i in range(N):
            errors[i] = get_error(i, gram_K, y, alphas, b)

        a_i, x_i, y_i, E_i = alphas[i], X[i], y[i], errors[i]
        a_j, x_j, y_j, E_j = alphas[j], X[j], y[j], errors[j]

        K_ii, K_jj, K_ij = np.dot(x_i, x_i), np.dot(x_j, x_j), np.dot(x_i, x_j)
        eta = K_ii + K_jj - 2 * K_ij
        if eta <= 0:
            return 0, b

        a_i_old, a_j_old = a_i, a_j
        a_j_new = a_j_old + y_j * (E_i - E_j) / eta

        # 对alpha进行修剪
        if y_i != y_j:
            L = max(0, a_j_old - a_i_old)
            H = min(C, C + a_j_old - a_i_old)
        else:
            L = max(0, a_i_old + a_j_old - C)
            H = min(C, a_j_old + a_i_old)

        a_j_new = clip(a_j_new, L, H)
        a_i_new = a_i_old + y_i * y_j * (a_j_old - a_j_new)

        if abs(a_j_new - a_j_old) < 0.00001:
            #print('WARNING   alpha_j not moving enough')
            return 0, b

        alphas[i], alphas[j] = a_i_new, a_j_new
        for i in range(N):
            errors[i] = get_error(i, gram_K, y, alphas, b)

        # 更新阈值b
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

        return 1, b
    else:
        return 0, b


class SVM(object):

    def __init__(self,
                 C=1,
                 max_iter=500,
                 tolerance=0.001,
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
        self.tolerance = tolerance

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

        self.class_num = np.max(y) + 1  # 最大的类
        assert self.class_num >= 2, 'class number should be larger than 2'
        # create kernel
        gamma = self.gamma
        if gamma == "scale":
            gamma = 1 / (X.shape[1] * X.var())
        elif gamma == 'auto':
            gamma = 1 / X.shape[1]
        self._gamma = gamma

        print(f'kernel start calc {datetime.now()}')
        gram_K = self._kernel_matrix(X)  # numba speed 6.46 => 1.54
        print(f'kernel end calc {datetime.now()}')

        if self.class_num == 2:
            y = np.where(y <= 0, -1, 1)  # 标签转换为-1和1
            # print(f'start calc {datetime.now()}')
            # numba speed 60.12 => 24.76
            alphas, X, y, b = self._fit(X, y, gram_K, self.max_iter, self.C,
                                        self.tolerance)
            # print(f'end calc {datetime.now()}')
            self.append_result(alphas, X, y, b)
        else:
            """
            multi class
            """
            y = np.eye(self.class_num)[y]
            y = np.where(y <= 0, -1, 1)  # 标签转换为-1和1
            for i in range(self.class_num):
                print(f'start calc {datetime.now()}')
                alphas, X_, y_, b = self._fit(X, y[:,
                                                   i], gram_K, self.max_iter,
                                              self.C, self.tolerance)
                print(f'end calc {datetime.now()}')
                self.append_result(alphas, X_, y_, b)
                if verbose:
                    print(f"trained class{i}")

        return self

    @staticmethod
    @jit(nopython=True)
    def _fit(X, y, gram_K, max_iter, C, tolerance):
        N, _ = X.shape

        alphas = np.zeros(N)
        b = 0
        # Cached errors ,f(x_i) - y_i
        errors = [get_error(i, gram_K, y, alphas, b) for i in range(N)]
        it = 0

        # 遍历所有alpha的标记
        entire = True

        pair_changed = 0
        while (it < max_iter):  #and (pair_changed > 0 or entire):
            pair_changed = 0
            if entire:
                for i in range(N):
                    pc, b = examine_example(i, alphas, gram_K, errors, X, y,
                                            tolerance, C, N, b)
                    pair_changed += pc
            else:
                non_bound_indices = [
                    i for i in range(N) if alphas[i] > 0 and alphas[i] < C
                ]
                for i in non_bound_indices:
                    pc, b = examine_example(i, alphas, gram_K, errors, X, y,
                                            tolerance, C, N, b)
                    pair_changed += pc
            it += 1

            if entire:
                entire = False
            elif pair_changed == 0:
                entire = True

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
import numpy as np
import math

"""
Solving closed BCMP network parameters
- pages 362, 463-464 Queueing Networks and Markov Chains, Bolch et al.
- http://home.agh.edu.pl/~kwiecien/metoda%20SUM.pdf

    assumptions:
        no class transitions,
        load independency,
        same service rate for all classes in node

Tested only for python 2.7
"""


class BcmpNetworkClosed(object):
    def __init__(self, R, N, k, mi_matrix, p, m, types, epsilon):
        self.R = R
        self.N = N
        if len(k) != R:
            raise ValueError("Amount of request types greater than amount of classes (%s, %s)" % (len(k), R))
        self.k = k
        self.k_sum = sum(k)
        if mi_matrix.shape != (self.N, self.R):
            raise ValueError("mi matrix should be shaped (%s, %s)" % (self.N, self.R))
        self.mi_matrix = mi_matrix
        # store servers amount per node (needed for ro recalculations)
        self.m = m
        if len(self.m) != self.N:
            raise ValueError("Incorrect length of server amounts list (%s != %s)" % (len(self.m), self.N))
        self.types = types
        if len(self.types) != self.N:
            raise ValueError("Incorrect length of node types list (%s != %s)" % (len(self.types), self.N))
        self.epsilon = epsilon
        # raw probabilites have to be converted
        self.e = self._calculate_visit_ratios(p)
        if self.e.shape != (self.N, self.R):
            raise ValueError("e matrix calculation failed: dimension mismatch (%s, %s)" % self.e.shape)
        # Initate lambdas with zeros
        self.lambdas = np.array([epsilon for _ in range(self.R)])

    def _calculate_visit_ratios(self, p):
        tempMatrix = np.zeros((self.N, self.N))
        row_list = []
        for i in range(0, len(p)):
            matList = []
            for j in range(0, len(p)):
                if i == j:
                    matList.append(p[i])
                else:
                    matList.append(tempMatrix)
            row = np.concatenate(matList)
            row_list.append(row)
        finishedMatrix = np.column_stack(row_list)

        b = ([1] + [0] * (self.N - 1)) * self.R
        a_minus = ([0] + [1] * (self.N - 1)) * self.R

        A = finishedMatrix.T - np.diagflat(a_minus)
        ret, _, _, _ = np.linalg.lstsq(A, b)

        visit_ratios = ret.reshape(self.R, self.N).T
        return visit_ratios

    def calculate_ro(self, i, r):
        mi = self.mi_matrix[i, r]

        if mi == 0 or self.types[i] == 3:
            return 0

        return self.lambdas[r] * self.e[i, r] / (self.m[i] * self.mi_matrix[i, r])

    def calculate_roi(self, i):
        return sum([self.calculate_ro(i, r) for r in range(self.R)])

    def calculate_pmi(self, i, roi):
        m = self.m[i]
        if m == 0:
            return 1
        elif roi == 0:
            return 0

        mul1 = ((m * roi) ** m) / (math.factorial(m) * (1 - roi))
        den1 = sum([((m * roi) ** k) / math.factorial(k) for k in range(m)])
        den2 = (((m * roi) ** m) / math.factorial(m)) * (1. / (1 - roi))

        return mul1 / (den1 + den2)

    def _type11(self, i, r, roi):
        mi = self.mi_matrix[i, r]
        if mi == 0:
            return 0

        nom = self.e[i, r] / mi
        denom = (1. - ((self.k_sum - 1.) / self.k_sum) * roi)

        return nom / denom

    def _type1n(self, i, r, roi, m):
        if self.mi_matrix[i, r] == 0:
            return 0

        sum1 = self.e[i, r] / self.mi_matrix[i, r]
        mul1 = (self.e[i, r] / (m * self.mi_matrix[i, r])) / (
            1. - (((self.k_sum - m - 1.) / (self.k_sum - m)) * roi))

        return sum1 + mul1 * self.calculate_pmi(i, roi)

    def _type31(self, i, r):
        return self.e[i, r] / self.mi_matrix[i, r]

    def _get_fix(self, i, r, roi):
        type_ = self.types[i]
        m = self.m[i]

        if m == 1 and type_ in frozenset([1, 2, 4]):
            return self._type11(i, r, roi)
        elif type_ == 1 and m > 1:
            return self._type1n(i, r, roi, m)
        elif m == 1 and type_ == 3:
            return self._type31(i, r)

        raise RuntimeError("Unsupported (type, amount) pair (%s, %s) " % (type_, m))

    def calculate_single_iteration(self):
        s = 0.
        for r in range(self.R):
            for i in range(self.N):
                roi = self.calculate_roi(i)
                fix = self._get_fix(i, r, roi)
                s += fix

            self.lambdas[r] = (float(self.k[r]) / s) if s != 0 else 0

    def calculate_lambdas_sum_method(self):
        error = self.epsilon + 1
        i = 0
        while error > self.epsilon:
            if i > 100:
                break

            old_lambdas = np.copy(self.lambdas)

            self.calculate_single_iteration()

            err = ((self.lambdas - old_lambdas) ** 2).sum()
            error = math.sqrt(err)
            i += 1

    def get_ro_matrix(self):
        ro_mt = np.zeros((8, 3))
        for r in range(self.R):
            for i in range(self.N):
                ro_mt[i, r] = self.calculate_ro(i, r)

        return ro_mt

    def get_k_matrix(self):
        k_mt = np.zeros((8, 3))
        for r in range(self.R):
            for i in range(self.N):
                if self.types[i] in [1,2,4] and self.m[i] == 1:
                    k_mt[i, r] = self.calculate_ro(i, r) / (1 - ((self.k_sum - 1) / self.k_sum)*self.calculate_roi(i))
                elif self.types[i] == 1 and self.m[i] > 1:
                    roi = self.calculate_roi(i)
                    roir = self.calculate_ro(i, r)
                    sum2 = roir / (1 - ((self.k_sum - self.m[i] - 1) / (self.k_sum - self.m[i])) * roi)
                    k_mt[i, r] = self.m[i] * roir + sum2 * self.calculate_pmi(i, roi)
                elif self.types[i] == 3 and self.m[i] == 1:
                    k_mt[i, r] = self.e[i, r] / self.mi_matrix[i, r]
        return k_mt

    def get_t_matrix(self):
        t_mt = np.zeros((8, 3))
        k_mt = self.get_k_matrix()
        for r in range(self.R):
            for i in range(self.N):
                lambda_ir = self.lambdas[r] * self.e[i, r]
                if self.e[i, r] == 0:
                    t_mt[i, r] = 0
                else:
                    t_mt[i, r] = k_mt[i, r] / lambda_ir
        return t_mt

    def get_w_matrix(self):
        w_mt = np.zeros((8, 3))
        t_mt = self.get_t_matrix()
        for r in range(self.R):
            for i in range(self.N):
                if self.types[i] == 3 or self.mi_matrix[i, r] == 0:
                    w_mt[i, r] = 0
                else:
                    w = t_mt[i, r] - 1 / self.mi_matrix[i, r]
                    w_mt[i, r] = w if w > 0 else 0
        return w_mt

    def get_measures(self):
        self.calculate_lambdas_sum_method()
        mean_w_matrix = self.get_w_matrix()
        return {
            'mean_w_matrix': mean_w_matrix
        }


def main():
    # Classes amount
    R = 3
    # Nodes amount
    N = 8

    # service times
    # mi = np.matrix([[8., 24.],
    #                 [12., 32.],
    #                 [16., 36.]])
    mi = np.array([[67., 67., 67.],
                    [8.,   8.,  8.],
                    [60., 60., 60.],
                    [8.33, 8.33, 8.33],
                    [12., 12., 12.],
                    [0.218, 0.218, 0.218],
                    [1.,  1.,  1.],
                    [0.92, 0.137, 0.053]])


    # single class transition probability matrices
    # matrix[node1,node2] denotes transition probability
    # from node1 to node2 for given class
    p1 = np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    p2 = np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    p3 = np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.2],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    # TODO check whether matrix dimensions are NxR

    # As there's no class transitions we can safely use list of
    # single class transition probabilities
    classes = [p1, p2, p3]

    # generate network parameters
    # node types
    types = [1, 3, 1, 1, 1, 1, 1, 3]
    # servers amount in respect to node type
    m = [1, 1, 1, 4, 2, 66, 30, 1]
    # amount of request by class
    K1 = [250, 144, 20]
    epsilon = 1e-05

    solver1 = BcmpNetworkClosed(
        R=R,
        N=N,
        k=K1,
        mi_matrix=mi,
        p=classes,
        m=m,
        types=types,
        epsilon=epsilon
    )

    solver1.calculate_lambdas_sum_method()
    print 'lambdar'
    print solver1.lambdas
    print 'e'
    print solver1.e
    print 'ro'
    print solver1.get_ro_matrix()
    print 'k'
    print solver1.get_k_matrix()
    print 't'
    print solver1.get_t_matrix()
    print 'w'
    print solver1.get_w_matrix()



if __name__ == '__main__':
    main()

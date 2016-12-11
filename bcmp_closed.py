import numpy as np
import random
import math
"""
Solving closed BCMP network parameters
- pages 362, 463-464 Queueing Networks and Markov Chains, Bolch et al.
- http://home.agh.edu.pl/~kwiecien/metoda%20SUM.pdf

    assumptions:
        no class transitions,
        load independency,
        same service rate for all classes in node

"""


def null(A, eps=1e-15):
    u, s, vh = np.linalg.svd(A)
    null_mask = (s <= eps)
    null_space = np.compress(null_mask, vh, axis=0)
    return np.transpose(null_space)


def nullspace(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


def get_ro_components(lambdas, visit_ratios, m, mi):
    ro_matrix = [[(lambdas[r] * visit_ratios[i, r]) / (m[i] * mi[i, r])
                  for r in range(R)] for i in range(N)]
    ro_matrix = np.matrix(ro_matrix)
    ro_list = ro_matrix.sum(1)
    # print(ro_matrix)
    # print(ro_list)
    return ro_matrix, ro_list


class Node(object):
    def __init__(self, type_, m):
        if type_ != 1 and m != 1:
            raise RuntimeError('Type %s can have only 1 server' % type_)
        self.type = type_
        self.m = m


class BcmpNetworkClosed(object):
    def __init__(self, R, N, k, mi_matrix, p, node_info, epsilon):
        self.R = R
        self.N = N
        self.k = k
        self.k_sum = sum(k)
        self.mi_matrix = mi_matrix
        self.m = list(map(lambda x: x[1], node_info))
        self.e = self._calculate_visit_ratios(p)
        self._lambdas = [0.00001 for _ in range(R)]
        self._get_call_chains(node_info)
        self.calculate_ro(self)
        self.epsilon = epsilon

    @staticmethod
    def _calculate_visit_ratios(p):
        visit_ratios = []
        for cl in p:
            A = cl.T - np.diagflat([1, 1, 1])
            visit_ratios.append(null(A))
        return -1 * np.hstack(visit_ratios)

    def calculate_ro(self):
        ro_matrix = [[(self._lambdas[r] * self.e[i, r]) / (self.m[i] * self.mi_matrix[i, r])
                      for r in range(R)] for i in range(N)]
        self.ro = ro_matrix.sum(1)

    def _get_call_chains(self, node_info):
        if any(map(lambda tup: tup[1] == 0, node_info)):
            raise ValueError('0 amount of servers is not allowed')
        if any(map(lambda (type, m): type !=1 and m != 1, node_info)):
            raise ValueError('Only type 1 can have more than one server')

        self.call_chain_matrix = [[] for _ in range(N)]
        for i in range(self.N):
            for r in range(self.R):
                self.call_chain_matrix[i].append(
                    self.get_fix_iteration(r, i, node_info[i]))

    def get_fix_iteration(self, r, i, node_info):
        type_, m = node_info

        def type1():
            nom = self.e[i][r] / self.mi_matrix[i][r]
            denom = (1 - ((self.k_sum - 1) / self.k)*self.ro_list[i])
            return nom / denom

        def type2():
            sum1 = self.e[i][r] / self.mi_matrix[i][r]
            mul1 = (sum1 / m) / (1 - (self.k_sum-m-1)*self.ro[i]/self.k_sum-m)
            mul2 = ((m * self.ro[i])**m) / (math.factorial(m) * (1 - self.ro[i]))
            mul31 = sum([((m*self.ro[i])**k)/math.factorial(k) for k in range(m - 1)])
            mul32 = ((m * self.ro[i])**m) / math.factorial(m) * (1-1/(1-self.ro[i]))

            return sum1 + mul1 * mul2 / (mul31 + mul32)

        def type3():
            return self.e[i][r] / self.mi_matrix[i][r]

        if m == 1 and type_ in frozenset([1,2,4]):
            return type1
        elif m != 1 and type_ == 1:
            return type2
        elif type_ == 3:
            return type3

    def iterate(self):
        error = self.epsilon + 1
        while error > self.epsilon:
            old_lambdas = list(self._lambdas)
            for r in enumerate(self._lambdas):
                self._lambdas[r] = self.k[r] / sum(map(lambda x: x(), self.call_chain_matrix[i]))
                err = sum([(self._lambdas[rr] - old_lambdas[rr])**2 for rr in enumerate(self._lambdas)])
                error = math.sqrt(err)
                self.calculate_ro()

if __name__ == '__main__':
    # Classes amount
    R = 2
    # Nodes amount
    N = 3

    # service times
    mi = np.matrix([[8., 24],
                    [12, 32],
                    [16, 36]])

    # single class transition probability matrices
    # matrix[node1,node2] denotes transition probability
    # from node1 to node2 for given class
    p1 = np.matrix([[0.0, 0.7, 0.3],
                    [0.6, 0.0, 0.4],
                    [0.5, 0.5, 0.0]])
    p2 = np.matrix([[0.0, 0.4, 0.6],
                    [0.7, 0.0, 0.3],
                    [0.4, 0.6, 0.0]])
    # TODO check whether matrix dimensions are NxR

    # As there's no class transitions we can safely use list of
    # single class transition probabilities
    classes = [p1, p2]

    # step 1 - compute visit ratios for closed network
    visit_ratios = []
    for cl in classes:
        from pprint import pprint
        A = cl.T - np.diagflat([1, 1, 1])
        # pprint(A)
        visit_ratios.append(null(A))
        # pprint(visit_ratio)
        # pprint(A*visit_ratio)
    # TODO why those bloody coeffs are negative
    visit_ratios = -1 * np.hstack(visit_ratios)

    # TODO make fixed parameters to fully test that
    # generate network parameters
    # node types
    types = [random.randint(1, 4) for _ in range(N)]
    # servers amount in respect to node type
    m = [random.randint(1, 4) if types[i] == 1 else 1 for i in range(len(types))]
    # amount of request by class
    K = [random.randint(1, 4) for _ in range(R)]

    # initialize lambdas
    lambdas = np.array([0.00001 for _ in range(R)])

    # k summed
    k_sum = sum(K)

    # fixed point iterations
    epsilon = 0
    e = epsilon + 1
    while e > epsilon:
        ro_matrix, roi_list = get_ro_components(lambdas, visit_ratios, m, mi)



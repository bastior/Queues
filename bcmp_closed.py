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
        """
        :param R: classes amount
        :param N: nodes amount
        :param k: list of request in system per class
        :param mi_matrix: matrix[node,class]
        :param p: list of matrices, probabilities for each class
        :param node_info: list of tuples (type, servers amount)
        :param epsilon: stop condition
        """
        self.R = R
        self.N = N
        self.k = k
        self.k_sum = sum(k)
        self.mi_matrix = mi_matrix
        # store servers amount per node (needed for ro recalculations)
        # im too lazy to clean that up
        self.m = list(map(lambda x: x[1], node_info))
        self.epsilon = epsilon
        # raw probabilites have to be converted
        self.e = self._calculate_visit_ratios(p)
        # Initate lambdas with zeros
        self._lambdas = [0.00001 for _ in range(R)]
        # determine function types and store closures for each
        self._get_call_chains(node_info)
        # calculate ro values pre first iteration
        self.calculate_ro()

    @staticmethod
    def _calculate_visit_ratios(p):
        visit_ratios = []
        for cl in p:
            A = cl.T - np.diagflat([1, 1, 1])
            visit_ratios.append(null(A))
        return -1 * np.hstack(visit_ratios)

    def calculate_ro(self):
        """
            Ro calculation without explicit lambda matrix, based on
            lambda_ir = e_ir * lambda_r
        :return:
        """
        ro_matrix = [[(self._lambdas[r] * self.e[i, r]) / (self.m[i] * self.mi_matrix[i, r])
                                for r in range(R)] for i in range(N)]
        ro_matrix = np.matrix(ro_matrix)
        self.ro = ro_matrix.sum(1)

    def _get_call_chains(self, node_info):
        """
        Populates call matrix with closures, a bit different
        way to determine computations based on node type.
        Implies heavy usage of class state.
        :param node_info: list of tuples from init
        :return: none
        """
        if any(map(lambda tup: tup[1] == 0, node_info)):
            raise ValueError('0 amount of servers is not allowed')
        if any(map(lambda (type, m): type !=1 and m != 1, node_info)):
            raise ValueError('Only type 1 can have more than one server')

        self.call_chain_matrix = [[] for _ in range(R)]
        for r in range(self.R):
            for i in range(self.N):
                self.call_chain_matrix[r].append(
                    self.get_fix_iteration(r, i, node_info[i]))

    def get_fix_iteration(self, r, i, node_info):
        """
        Lets hope all equations are valid. No guarantees given for now.
        :param r: class index
        :param i: node index
        :param node_info: tuple(type, servers amount)
        :return: function calculating fix for (type, node, class) scheme
        """
        type_, m = node_info

        def type1():
            nom = self.e[i,r] / self.mi_matrix[i,r]
            denom = (1 - ((self.k_sum - 1) / self.k_sum) * self.ro[i]).item()
            return nom / denom

        def type2():
            sum1 = self.e[i, r] / self.mi_matrix[i, r]
            mul1 = (sum1 / m) / (1 - (self.k_sum-m-1)*self.ro[i]/self.k_sum-m)
            mul2 = ((m * self.ro[i])**m) / (math.factorial(m) * (1 - self.ro[i]))
            mul31 = sum([((m*self.ro[i])**k)/math.factorial(k) for k in range(m - 1)])
            mul32 = ((m * self.ro[i])**m) / math.factorial(m) * (1-1/(1-self.ro[i]))

            return (sum1 + mul1 * mul2 / (mul31 + mul32)).item()

        def type3():
            return self.e[i, r] / self.mi_matrix[i, r]

        if m == 1 and type_ in frozenset([1,2,4]):
            return type1
        elif m != 1 and type_ == 1:
            return type2
        elif type_ == 3:
            return type3

        raise RuntimeError("Unsupported (type, amount) pair (%s, %s) " % node_info)

    def iterate(self):
        error = self.epsilon + 1
        while error > self.epsilon:
            old_lambdas = list(self._lambdas)
            for r in range(R):
                self._lambdas[r] = self.k[r] / sum(map(lambda x: x(), self.call_chain_matrix[r]))

            err = sum([(self._lambdas[r] - old_lambdas[r])**2 for r in range(R)])
            error = math.sqrt(err)
            self.calculate_ro()

    def get_measures(self):
        pass


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

    # generate network parameters
    # node types
    types = [1,1,3]
    # servers amount in respect to node type
    m = [2,1,1]
    # amount of request by class
    K = [3,3,3]

    # Raise solver
    solver = BcmpNetworkClosed(
        R=R,
        N=N,
        k=K,
        mi_matrix=mi,
        p=classes,
        node_info=zip(types,m),
        epsilon=0.0001
    )

    solver.iterate()


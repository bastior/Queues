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

Tested only for python 2.7
"""


def get_ro_components(lambdas, visit_ratios, m, mi):
    ro_matrix = [[(lambdas[r] * visit_ratios[i, r]) / (m[i] * mi[i, r])
                  for r in range(self.R)] for i in range(self.N)]
    ro_matrix = np.matrix(ro_matrix)
    ro_list = ro_matrix.sum(1)
    # print(ro_matrix)
    # print(ro_list)
    return ro_matrix, ro_list


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
        if len(k) != R:
            raise ValueError("Amount of request types greater than amount of classes (%s, %s)" % (len(k), R))
        self.k = k
        self.k_sum = sum(k)
        self.mi_matrix = mi_matrix
        # store servers amount per node (needed for ro recalculations)
        # im too lazy to clean that up
        self.m = list(map(lambda x: x[1], node_info))
        # this should be split, laziness emerges
        self.types = list(map(lambda x: x[0], node_info))
        self.epsilon = epsilon
        # raw probabilites have to be converted
        self.e = self._calculate_visit_ratios(p)
        # Initate lambdas with zeros
        self._lambdas = [0.00001 for _ in range(self.R)]
        # determine function types and store closures for each
        self._get_call_chains(node_info)
        # calculate ro values pre first iteration
        self._recalculate_lambda_ir()
        self._calculate_ro()

    @staticmethod
    def _calculate_visit_ratios(p):
        visit_ratios = []
        for cl in p:
            A = cl.T - np.diagflat([0, 1, 1])
            ret = np.linalg.solve(A, [1, 0, 0])
            visit_ratios.append(ret)
        return np.vstack(visit_ratios).T

    def _recalculate_lambda_ir(self):
        lambda_matrix = [[(self._lambdas[r] * self.e[i, r]) for r in range(self.R)] for i in range(self.N)]
        self.lambda_matrix = np.matrix(lambda_matrix)

    def _calculate_ro(self):
        """
            Ro calculation without explicit lambda matrix, based on
            lambda_ir = e_ir * lambda_r
        :return:
        """
        ro_matrix = [[self.lambda_matrix[i, r] / (self.m[i] * self.mi_matrix[i, r])
                      for r in range(self.R)] for i in range(self.N)]
        self.ro_matrix = np.matrix(ro_matrix)
        self.ro = self.ro_matrix.sum(1)

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
        if any(map(lambda (type, m): type != 1 and m != 1, node_info)):
            raise ValueError('Only type 1 can have more than one server')

        self.call_chain_matrix = [[] for _ in range(self.R)]
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
            nom = self.e[i, r] / self.mi_matrix[i, r]
            denom = (1 - ((self.k_sum - 1) / self.k_sum) * self.ro[i]).item()
            return nom / denom

        def type2():
            sum1 = self.e[i, r] / self.mi_matrix[i, r]
            mul1 = (self.e[i, r] / (m * self.mi_matrix[i, r])) / (
                1 - self.ro[i] * ((self.k_sum - m - 1) / (self.k_sum - m)))
            return (sum1 + mul1 * self.calculate_pmi(i)).item()

        def type3():
            return self.e[i, r] / self.mi_matrix[i, r]

        if m == 1 and type_ in frozenset([1, 2, 4]):
            return type1
        elif m != 1 and type_ == 1:
            return type2
        elif type_ == 3:
            return type3

        raise RuntimeError("Unsupported (type, amount) pair (%s, %s) " % node_info)

    def calculate_pmi(self, i):
        m = self.m[i]
        mul1 = ((m * self.ro[i]) ** m) / (math.factorial(m) * (1 - self.ro[i]))
        den1 = sum([((m * self.ro[i]) ** k) / math.factorial(k) for k in range(m - 1)])
        den2 = (((m * self.ro[i]) ** m) / math.factorial(m)) * (1 / (1 - self.ro[i]))
        return mul1 / (den1 + den2)

    def _iterate(self):
        error = self.epsilon + 1
        while error > self.epsilon and not self._validate_lambdas():
            old_lambdas = list(self._lambdas)
            for r in range(self.R):
                vals = map(lambda x: x(), self.call_chain_matrix[r])
                s = sum(vals)
                new_lambda = self.k[r] / s
                self._lambdas[r] = new_lambda

            err = sum([(self._lambdas[r] - old_lambdas[r]) ** 2 for r in range(self.R)])
            error = math.sqrt(err)
            self._recalculate_lambda_ir()
            self._calculate_ro()

    def get_kri(self, i, r):
        m = self.m[i]
        type_ = self.types[i]
        if type_ in frozenset([1, 2, 4]) and m == 1:
            return (self.ro_matrix[i, r] / (1 - self.ro[i] * (self.k_sum - 1) / self.k_sum)).item()
        elif type_ == 1 and m > 1:
            ro_ir = self.ro_matrix[i, r]
            return (m * ro_ir + self.calculate_pmi(i) * ro_ir / (
                1 - self.ro[i] * (self.k_sum - m - 1) / (self.k_sum - m))).item()
        elif type_ == 3:
            return self._lambdas[r] * self.e[i, r] / self.mi_matrix[i, r]
        raise RuntimeError("Unsupported (type, amount) pair (%s, %s) " % (m, type_))

    def _validate_lambdas(self):
        flag = True
        for i in range(self.N):
            for r in range(self.R):
                rhs = self.m[i] * self.mi_matrix[i, r]
                if self.lambda_matrix[i, r] > rhs:
                    flag &= False

        return flag

    def get_measures(self):
        self._iterate()
        mean_k_matrix = [[self.get_kri(i, r) for r in range(self.R)] for i in range(self.N)]
        mean_t_matrix = [[mean_k_matrix[i][r] / (self.e[i, r] * self._lambdas[r]) for r in range(self.R)] for i in
                         range(self.N)]
        mean_w_matrix = [[mean_t_matrix[i][r] - (1 / self.mi_matrix[i, r]) for r in range(self.R)] for i in
                         range(self.N)]
        return {
            'mean_k_matrix': mean_k_matrix,
            'mean_t_matrix': mean_t_matrix,
            'mean_w_matrix': mean_w_matrix
        }


if __name__ == '__main__':
    # Classes amount
    R = 2
    # Nodes amount
    N = 3

    # service times
    # mi = np.matrix([[8., 24.],
    #                 [12., 32.],
    #                 [16., 36.]])
    mi = np.matrix([[0.01, 0.04],
                    [0.02, 0.05],
                    [0.03, 0.06]])

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
    types = [1, 1, 1]
    # servers amount in respect to node type
    m = [2, 1, 1]
    # amount of request by class
    K1 = [1, 2]
    K2 = [100000000, 200000000]

    # initiate solver
    solver1 = BcmpNetworkClosed(
        R=R,
        N=N,
        k=K1,
        mi_matrix=mi,
        p=classes,
        node_info=zip(types, m),
        epsilon=0.0001
    )

    solver2 = BcmpNetworkClosed(
        R=R,
        N=N,
        k=K2,
        mi_matrix=mi,
        p=classes,
        node_info=zip(types, m),
        epsilon=0.0001
    )

    from pprint import pprint

    res1 = solver1.get_measures()
    res2 = solver2.get_measures()

    W1 = np.matrix(res1['mean_w_matrix'])
    W2 = np.matrix(res2['mean_w_matrix'])
    pprint(W2 - W1)
    # pprint(res)

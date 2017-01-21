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
        if mi_matrix.shape != (self.N, self.R):
            raise ValueError("mi matrix should be shaped (%s, %s)" % (self.N, self.R))
        self.mi_matrix = mi_matrix
        # store servers amount per node (needed for ro recalculations)
        # im too lazy to clean that up
        self.m = np.array(map(lambda x: x[1], node_info))
        if len(self.m) != self.N:
            raise ValueError("Incorrect length of server amounts list (%s != %s)" % (len(self.m), self.N))
        # this should be split, laziness emerges
        self.types = list(map(lambda x: x[0], node_info))
        if len(self.types) != self.N:
            raise ValueError("Incorrect length of node types list (%s != %s)" % (len(self.types), self.N))
        self.epsilon = epsilon
        # raw probabilites have to be converted
        self.e = self._calculate_visit_ratios(p)
        if self.e.shape != (self.N, self.R):
            raise ValueError("e matrix calculation failed: dimension mismatch (%s, %s)" % self.e.shape)
        # Initate lambdas with zeros
        self._lambdas = np.array([0.00001 for _ in range(self.R)])
        # determine function types and store closures for each
        self._get_call_chains(node_info)
        # calculate ro values pre first iteration
        self._recalculate_lambda_ir()
        self._calculate_ro()

    def _calculate_visit_ratios(self, p):
        visit_ratios = []
        tempMatrix = np.zeros((self.N, self.N))
        # finishedMatrix = np.zeros((len(p[0]) * len(p) , 0))
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

    def _recalculate_lambda_ir(self):
        self.lambda_matrix = self._lambdas * self.e

    def _calculate_ro(self):
        """
            Ro calculation without explicit lambda matrix, based on
            lambda_ir = e_ir * lambda_r
        :return:
        """
        self.ro_matrix = self.lambda_matrix / (self.mi_matrix * self.m.reshape(self.N, 1))
        self.ro = self.ro_matrix.sum(1)
        # print self.ro

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
        if any(map(lambda (type_, m): type_ != 1 and m != 1, node_info)):
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
            sum1 = self.e[i, r] / self.mi_matrix[i, r]
            mul1 = (self.e[i, r] / (m * self.mi_matrix[i, r])) / (
                1. - self.ro[i] * ((self.k_sum - m - 1.) / (self.k_sum - m)))
            return (sum1 + mul1 * self.calculate_pmi(i))

        def type2():
            nom = self.e[i, r] / self.mi_matrix[i, r]
            denom = (1. - ((self.k_sum - 1.) / self.k_sum) * self.ro[i])
            # print 'type2'
            # print '[%s, %s]' % (i+1, r+1)
            # print self.e[i, r]
            # print self.mi_matrix[i, r]
            # print self.ro[i]
            # print nom/denom
            # print
            return nom / denom

        def type3():
            return self.e[i, r] / self.mi_matrix[i, r]

        if type_ == 1 and m > 1:
            return type1
        elif m == 1 and type_ == 3:
            return type3
        elif m == 1 and type_ in frozenset([1, 2, 4]):
            return type2

        raise RuntimeError("Unsupported (type, amount) pair (%s, %s) " % node_info)

    def calculate_pmi(self, i):
        m = self.m[i]
        mul1 = ((m * self.ro[i]) ** m) / (math.factorial(m) * (1 - self.ro[i]))
        den1 = sum([((m * self.ro[i]) ** k) / math.factorial(k) for k in range(m - 1)])
        den2 = (((m * self.ro[i]) ** m) / math.factorial(m)) * (1. / (1 - self.ro[i]))
        return mul1 / (den1 + den2)

    def _iterate(self):
        error = self.epsilon + 1
        i = 1
        while error > self.epsilon:
            old_lambdas = np.copy(self._lambdas)
            # print i
            i += 1
            s = 0
            for r in range(self.R):
                vals = [fun() for fun in self.call_chain_matrix[r]]
                s += sum(vals)
                new_lambda = self.k[r] / s
                self._lambdas[r] = new_lambda

            err = ((self._lambdas - old_lambdas) ** 2).sum()
            error = math.sqrt(err)
            self._recalculate_lambda_ir()
            self._calculate_ro()

    def get_kri(self, i, r):
        m = self.m[i]
        type_ = self.types[i]
        if type_ in frozenset([2, 4]) and m == 1:
            return self.ro_matrix[i, r] / (1 - self.ro[i] * (self.k_sum - 1) / self.k_sum)
        elif type_ == 1 and m >= 1:
            ro_ir = self.ro_matrix[i, r]
            result = m * ro_ir + (ro_ir / (1 - self.ro[i])) * self.calculate_pmi(i)
            return result
        elif type_ == 3 and m == 1:
            return self.lambda_matrix[i, r] / self.mi_matrix[i, r]
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
        mean_t_matrix = [[0] * self.R] * self.N
        mean_w_matrix = [[0] * self.R] * self.N
        for r in range(self.R):
            for i in range(self.N):
                if self.lambda_matrix[i, r] != 0:
                    mean_t_matrix[i][r] = mean_k_matrix[i][r] / self.lambda_matrix[i, r]
                    mean_w_matrix[i][r] = mean_t_matrix[i][r] - (1. / self.mi_matrix[i, r])

        return {
            'mean_k_matrix': mean_k_matrix,
            'mean_t_matrix': mean_t_matrix,
            'mean_w_matrix': mean_w_matrix
        }


class BcmpNetworkClosedV2(object):
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
        den1 = sum([((m * roi) ** k) / math.factorial(k) for k in range(m - 1)])
        den2 = (((m * roi) ** m) / math.factorial(m)) * (1. / (1 - roi))

        return mul1 / (den1 + den2)

    def _type11(self, i, r, roi):
        mi = self.mi_matrix[i, r]
        if mi == 0:
            return 0

        nom = self.e[i, r] / mi
        denom = (1. - ((self.k_sum - 1.) / self.k_sum) * roi)
        print i+1, r+1, self.e[i, r], self.mi_matrix[i, r], roi, nom/denom

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

    def get_params_sum_method(self):
        error = self.epsilon + 1
        # i = 0
        while error > self.epsilon:
            # if i > 100:
            #     break

            old_lambdas = np.copy(self.lambdas)

            self.calculate_single_iteration()

            err = ((self.lambdas - old_lambdas) ** 2).sum()
            error = math.sqrt(err)
            # i += 1

    def get_ro_matrix(self):
        ro_mt = np.zeros((8, 3))
        for r in range(self.R):
            for i in range(self.N):
                ro_mt[i, r] = self.calculate_ro(i, r)

        return ro_mt

    def get_k_matrix(self):
        ro_mt = np.zeros((8, 3))
        for r in range(self.R):
            for i in range(self.N):
                ro_mt[i, r] = self.calculate_ro(i, r)



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

    solver1 = BcmpNetworkClosedV2(
        R=R,
        N=N,
        k=K1,
        mi_matrix=mi,
        p=classes,
        m=m,
        types=types,
        epsilon=0.0001
    )

    solver1.get_params_sum_method()
    # print solver1.lambdas
    # print solver1.e
    # print solver1.get_ro_matrix()

    # initiate solver
    solver2 = BcmpNetworkClosed(
        R=R,
        N=N,
        k=K1,
        mi_matrix=mi,
        p=classes,
        node_info=zip(types, m),
        epsilon=0.0001
    )
    res2 = solver2.get_measures()
    print solver2._lambdas


    # res1 = solver1.get_measures()
    # W1 = np.matrix(res1['mean_w_matrix'])
    # print solver1.e
    # print solver1._lambdas
    # from pprint import pprint
    # pprint(solver1.call_chain_matrix)
    # print solver1.lambda_matrix
    # print 'ro'
    # print solver1.ro_matrix
    # print 'mean_k'
    # print np.matrix(res1['mean_k_matrix'])
    # print 'mean_t'
    # print np.matrix(res1['mean_t_matrix'])
    # print 'mean_w'
    # print W1


if __name__ == '__main__':
    main()

import numpy as np
import math
import itertools


class QueueSystem:
    def __init__(self, lambda_list, u_list, qtype, **kwargs):
        lambda_len = len(lambda_list)
        u_len = len(u_list)
        if lambda_len != u_len:
            raise ValueError("Different amount of classes for u and lambda (%s, %s)" % (u_len, lambda_len))
        self.type = qtype
        self.lambdas = lambda_list
        self.u_list = u_list
        self.dict = kwargs

    def __repr__(self):
        return str("%s\n%s\n%s\n%s" % (self.type, self.lambdas, self.u_list, self.dict))


class QueueSystem2(object):
    cnt = itertools.count(0)

    def __init__(self, lambda_list, u_list):
        lambda_len = len(lambda_list)
        u_len = len(u_list)
        if lambda_len != u_len:
            raise ValueError("Different amount of classes for u and lambda (%s, %s)" % (u_len, lambda_len))
        self.lambdas = lambda_list
        self.u_list = u_list
        self.q_id = next(self.cnt)

    @staticmethod
    def _calculate_fifo_kri(m, lamb, u , roi):
        """
        :param m: channels amount in system
        :param lamb: service coefficient
        :param u: arrival coefficient
        :param roi: dunno, jest w pracy prowadzacej
        :return: average requests amount in system
        """
        ro = lamb / (m * u)
        mroi = m * roi
        fact = 1. / math.factorial(m)
        inv = 1. / (1. - roi)

        pow_mroi = mroi ** m

        den1 = sum([mroi ** k / math.factorial(k) for k in range(m - 1)])
        den2 = pow_mroi * fact * inv

        return m * ro + inv ** 2 * ro * pow_mroi * inv * 1 / (den1 + den2)

    @staticmethod
    def _calculate_ir_kri(lamb, u):
        """
        :param lamb: service coefficient
        :param u: arrival coefficient
        :return: average request amount in system
        """
        return lamb / u

    @staticmethod
    def _get_system_parameters(kir, lambir, uir):
        """
        :param kir: service time
        :param lambir: arrival coefficient
        :param uir: service coefficient
        :return: dict with common parameters
        """
        service_time = kir
        average_time_in_queue = kir/lambir
        average_queue_length = 0

        return {
            'service_time': service_time,
            'average_entries_count': average_entries_count,
            'average_queue_length': average_queue_length
        }

    def __repr__(self):
        return "%s: {lambdas: %s, u_list: %s}" % (self.q_id, self.lambdas, self.u_list)


class FifoQueueSystem(QueueSystem2):
    def __init__(self, m, **kwargs):
        super(FifoQueueSystem, self).__init__(**kwargs)
        self.m = m
        self.roi = sum([self.lambdas[k]/(self.m * self.u_list[k]) for k in range(len(self.u_list))])

    def __repr__(self):
        return "super: {%s}, m: %s, roi: %s" % (super(FifoQueueSystem, self).__repr__(), self.m, self.roi)


class QueueSolver:
    def __init__(self):
        pass


def get_bcmp_parameters(m_list, type_list, lamb_matrix, u_matrix):
    if not len(m_list) == len(lamb_matrix) == len(u_matrix):
        raise ValueError('Dimension mismatch')
    # TODO: ten warunek jest ubogi
    if len(lamb_matrix[0]) == 0 or len(u_matrix[0]) == 0:
        raise ValueError("Matrixes contain empty rows")
    if any([x == 0 for x in m_list]):
        raise ValueError("You need at least one channel in each queue")

    return _get_bcmp_parameters(m_list, type_list, lamb_matrix, u_matrix)


def _get_bcmp_parameters(m_list, type_list, lamb_matrix, u_matrix):
    pass


def calculate_kri(m_list, type_list, lamb_matrix, u_matrix):
    pass


def bcmp_ir(lamb, u):
    """
    :param lamb: service coefficient
    :param u: arrival coefficient
    :return: average request amount in system
    """
    return lamb / u


def bcmp_fifo(m, lamb, u, roi):
    """
    :param m: channels amount in system
    :param lamb: service coefficient
    :param u: arrival coefficient
    :param roi: dunno, jest w pracy prowadzacej
    :return: average requests amount in system
    """

    ro = lamb / (m * u)
    mroi = m * roi
    fact = 1. / math.factorial(m)
    inv = 1. / (1. - roi)

    pow_mroi = mroi**m

    den1 = sum([mroi**k/math.factorial(k) for k in range(m-1)])
    den2 = pow_mroi * fact * inv

    return m * ro + inv**2 * ro * pow_mroi * inv * 1 / (den1 + den2)


def main():
    # canals_amount_list = np.array([2, 1, 3])
    # type_array = [bcmp_fifo, bcmp_fifo, bcmp_ir]
    # service_time_coeffs_matrix = np.array([[2., 3.],
    #                                        [1., 1.],
    #                                        [3., 4.]])
    # user_arrival_coeffs_matrix = np.array([[3., 2.],
    #                                        [2., 3.],
    #                                        [2., 1.]])
    #
    # lamb0 = user_arrival_coeffs_matrix[0]
    # u0 = service_time_coeffs_matrix[0]
    # m0 = canals_amount_list[0]
    #
    # vals = [lamb0[k]/(m0 * u0[k]) for k in range(len(lamb0))]
    #
    # roi_0 = sum(vals)
    #
    # testval = bcmp_fifo(canals_amount_list[0],
    #                                service_time_coeffs_matrix[0][0],
    #                                user_arrival_coeffs_matrix[0][0],
    #                                roi_0)
    # print(testval)
    #
    # retval = get_bcmp_parameters(canals_amount_list, service_time_coeffs_matrix, user_arrival_coeffs_matrix)
    # print(retval)

    a = FifoQueueSystem(u_list=[1.,2.,3.], lambda_list=[3.,2.,1.], m=2)

if __name__ == '__main__':
    main()


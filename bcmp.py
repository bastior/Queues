import numpy as np
import math


def get_bcmp_parameters(canals_amount_list, service_time_coeffs_matrix, user_arrival_coeffs_matrix):
    if not len(canals_amount_list) == len(service_time_coeffs_matrix) == len(user_arrival_coeffs_matrix):
        raise ValueError('Dimension mismatch')
    # TODO: ten warunek jest ubogi
    if len(service_time_coeffs_matrix[0]) == 0 or len(user_arrival_coeffs_matrix[0]) == 0:
        raise ValueError("Matrixes contain empty rows")
    if any([x == 0 for x in canals_amount_list]):
        raise ValueError("You need at least one canal in each queue")

    return _get_bcmp_parameters(canals_amount_list, service_time_coeffs_matrix, user_arrival_coeffs_matrix)


def _get_bcmp_parameters(canals_amount_list, service_time_coeffs_matrix, user_arrival_coeffs_matrix):
    pass


def calculate_kri(canals_amount_list, service_time_coeffs_matrix, user_arrival_coeffs_matrix):
    pass


def calculate_single_kri(m, lamb, u, roi):
    """
    :param m: channels amount in system
    :param lamb: service coefficient
    :param u: arrival coefficient
    :param roi: dunno, jest w pracy prowadzacej
    :return: average requests amount in system
    """

    ro = lamb / (m * u)
    mroi = m * roi
    fact = 1 / math.factorial(m)
    inv = 1 / (1 - roi)

    pow_mroi = mroi**m

    den1 = sum([mroi**k/math.factorial(k) for k in range(m-1)])
    den2 = pow_mroi * fact * inv

    return m * ro + inv**2 * ro * pow_mroi * inv * 1 / (den1 + den2)


def main():
    canals_amount_list = np.array([2, 1, 3])
    service_time_coeffs_matrix = np.array([[2., 3.],
                                           [1., 1.],
                                           [3., 4.]])
    user_arrival_coeffs_matrix = np.array([[3., 2.],
                                           [2., 3.],
                                           [2., 1.]])

    lamb0 = user_arrival_coeffs_matrix[0]
    u0 = service_time_coeffs_matrix[0]
    m0 = canals_amount_list[0]

    vals = [lamb0[k]/(m0 * u0[k]) for k in range(len(lamb0))]

    roi_0 = sum(vals)

    testval = calculate_single_kri(canals_amount_list[0],
                                   service_time_coeffs_matrix[0][0],
                                   user_arrival_coeffs_matrix[0][0],
                                   roi_0)
    print(testval)

    retval = get_bcmp_parameters(canals_amount_list, service_time_coeffs_matrix, user_arrival_coeffs_matrix)
    print(retval)

if __name__ == '__main__':
    main()

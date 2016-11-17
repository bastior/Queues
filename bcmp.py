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


def calculate_single_kri(canals_amout, service_time_coeff, user_arrival_coeff):
    """
    :param canals_amout: m_i
    :param service_time_coeff: u_ir
    :param user_arrival_coeff: lambda_ir
    :return: k_ir value
    ro_ir = lambda_ir / ( m_i * u_ir )
    formula:
    k = m*ro+(ro/(1-ro))*(((m*ro)^m)/(m!*(1-ro)))*1/(sum ki = 0 to m-1:((m*ro)^ki)/(ki!)+(((m*ro)^m)/(m!))*(1/(1-ro)))
    K = m_ro + stat1*stat2*stat3
    """
    m = canals_amout
    ro = user_arrival_coeff/(canals_amout * service_time_coeff)
    m_ro = canals_amout * ro

    m_fact = math.factorial(m)
    stat1 = ro/(1 - ro)
    stat2 = m_ro**m / (m_fact * (1 - ro))
    stat3_static_part = (m_ro**m/m_fact)*(1/1-ro)

    stat3_denominator = 0
    for k in range(m-1):
        stat3_denominator += m_ro**k / math.factorial(k) + stat3_static_part
    stat3 = 1 / stat3_denominator

    return m_ro + stat1 * stat2 * stat3



def main():
    canals_amount_list = np.array([2, 1, 3])
    service_time_coeffs_matrix = np.array([[2., 3.],
                                           [1., 1.],
                                           [3., 4.]])
    user_arrival_coeffs_matrix = np.array([[3., 2.],
                                           [2., 3.],
                                           [2., 1.]])

    testval = calculate_single_kri(canals_amount_list[0], service_time_coeffs_matrix[0][0], user_arrival_coeffs_matrix[0][0])
    print(testval)

    retval = get_bcmp_parameters(canals_amount_list, service_time_coeffs_matrix, user_arrival_coeffs_matrix)
    print(retval)

if __name__ == '__main__':
    main()

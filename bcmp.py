import numpy as np

"""
Solving BCMP network parameters
- page 362 Queueing Networks and Markov Chains, Bolch et al.

    Model without class transitions - thus we generate
    node transition matrix separately for each request class
"""

if '__main__' == __name__:
    lambdas = np.array([1, 1])
    # service times per second [node, class]
    mi = np.matrix([[8., 24],
                    [12, 32],
                    [16, 36]])
    # [0,n] denotes entering n-th node from outside, analogous for [n,0]
    p1 = np.matrix([[0.0, 1.0, 0.0, 0.0],
                    [0.3, 0.0, 0.4, 0.3],
                    [0.0, 0.6, 0.0, 0.4],
                    [0.0, 0.5, 0.5, 0.0]])
    p2 = np.matrix([[0.0, 1.0, 0.0, 0.0],
                    [0.1, 0.0, 0.3, 0.6],
                    [0.0, 0.7, 0.0, 0.3],
                    [0.0, 0.4, 0.6, 0.0]])

    classes = [p1, p2]

    # compute visit ratios
    results = []
    for cl in classes:
        A = cl[1:, 1:].getT() - np.diagflat([1, 1, 1])
        b = cl[0, 1:].getT()
        results.append(-1 * A.I * b)

    visit_ratios = np.hstack(results)

    from pprint import pprint
    pprint(visit_ratios)
    # compute node utilisation matrix [node, class]
    old_numpy_state = np.seterr(divide='raise')
    node_util_matrix = visit_ratios.getA() * lambdas / mi.getA()
    node_util_vector = map(sum, node_util_matrix)
    pprint(node_util_vector)

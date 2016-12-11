import numpy as np
import random
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

if __name__ == '__main__':
    # Classes amount
    R = 3
    # Nodes amount
    N = 3

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

    # initialize lambdas
    lambdas = np.array([0.00001 for _ in range(R)])
    from pprint import pprint
    # pprint(tup)



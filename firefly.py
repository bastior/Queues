import math
import numpy as np
import json_tricks.np as jsont

from bcmp_closed import BcmpNetworkClosed


class FireflyAlgorithm(object):
    def __init__(self):
        self.config = jsont.load('config.json')
        self.ffa_config = self.config.get('ffa')
        self.bcmp_config = self.config.get('bcmp')

    def ffa(self, lower_bound, upper_bound, dim, n, max_generation):
        """
        lower_bound - LB of all values within firefly
        upper_bound - guess what....
        dim - dimension of firefly
        n - number of fireflies in algorithm
        max_generation - number of iterations that should be taken
        """

        alpha = 0.5
        beta_min = 0.2
        gamma = 1

        # value function for all fireflies. Bring to infinity, because why not
        zn = np.ones(n)
        zn.fill(float("inf"))

        # generate random fireflies between UB and LB
        ns = np.random.uniform(0, 1, (n, dim)) * (upper_bound - lower_bound) + lower_bound
        ns = ns.astype(int)

        light = np.ones(n)
        light.fill(float("inf"))

        # main loop
        for k in range(0, max_generation):

            for i in range(0, n):
                zn[i] = self.calc_value(ns[i])
                light[i] = zn[i]

            light = np.sort(zn)
            index = np.argsort(zn)
            ns = ns[index, :]

            nso = ns
            lightO = light
            nbest = ns[0, :]
            lightBest = light[0]

            fbest = lightBest

            scale = np.ones(dim) * abs(upper_bound - lower_bound)

            for i in range(0, n):
                # attractiveness parameter = exp(-gamma * r)
                for j in range(0, n):
                    r = np.sqrt(np.sum((ns[i, :] - ns[j, :]) ** 2))

                    if light[i] > lightO[j]:
                        beta0 = 1
                        beta = (beta0 - beta_min) * math.exp(-gamma * r ** 2) + beta_min
                        tmpf = alpha * (np.random.rand(dim) - 0.5) * scale
                        ns[i, :] = ns[i, :] * (1 - beta) + nso[j, :] * beta + tmpf
                        # sometimes it happens to be 0. Better safe then sorry
                        ns[i][ns[i] < 2] = 2

            iteration_number = k
            best_quality = fbest

            print "Iteration number:"
            print k
            print "Best value function:"
            print fbest
            print "for vector:"
            print nbest

    def calc_value(self, m):
        """
        """
        avr_times = self.bcmp_interface(m.tolist())
        time_sum = 0
        for i in avr_times:
            for j in i:
                time_sum += j

        if time_sum < 0:
            time_sum = 0

        function_value = 0

        for i in m:
            function_value += i * 0.01
        function_value += time_sum * 1000

        return function_value

    def bcmp_interface(self, m):
        types_list = self.bcmp_config.get('types')
        nr_of_class_one = sum(i == 1 for i in types_list)
        if nr_of_class_one != len(m):
            raise ValueError("Amount of ms must be equal to nodes with type = 1")

        final_m = [1 if type_val > 1 else m.pop(0) for type_val in types_list]

        solver = BcmpNetworkClosed(
            R=self.bcmp_config['R'],
            N=self.bcmp_config['N'],
            k=self.bcmp_config['k'],
            mi_matrix=self.bcmp_config['mi'],
            p=self.bcmp_config['classes'],
            types=self.bcmp_config['types'],
            m=final_m,
            epsilon=self.bcmp_config['epsilon']
        )
        vals = solver.get_measures()

        return vals['mean_w_matrix']


if __name__ == "__main__":
    m = [6, 4, 4]
    solution = FireflyAlgorithm()

    solution.ffa(**solution.ffa_config)
    print solution.bcmp_interface(m)

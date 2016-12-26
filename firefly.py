from bcmp_closed import *
import json_tricks.np as jsont


class FireflyAlgorithm(object):
    def __init__(self):
        self.config = jsont.load('config.json')
        self.ffa_config = self.config.get('ffa')
        self.bcmp_config = self.config.get('bcmp')

    def FFA(self, lowerBound, upperBound, dim, n, maxGeneration):
        """
        lowerBound - LB of all values within firefly
        upperBound - guess what....
        dim - dimention of firefly
        n - number of fireflies in algorythm
        maxGeneration - number of iterations that should be taken
        """


        alpha = 0.5
        betaMin = 0.2
        gamma = 1

        # value function for all fireflies. Bring to infinity, becouse why not
        zn = np.ones(n)
        zn.fill(float("inf"))

        #generate random fireflies between UB and LB
        ns = np.random.uniform(0,1,(n,dim)) * (upperBound - lowerBound) + lowerBound
        ns = ns.astype(int)

        light = np.ones(n)
        light.fill(float("inf"))

        #main loop
        for k in range(0,maxGeneration):

            for i in range(0,n):
                zn[i] = self.calcValue(ns[i])
                light[i] = zn[i]


            light = np.sort(zn)
            index = np.argsort(zn)
            ns = ns[index,:]

            nso = ns
            lightO = light
            nbest = ns[0,:]
            lightBest = light[0]

            fbest = lightBest

            scale = np.ones(dim) * abs(upperBound - lowerBound)

            for i in range (0,n):
                #attractivenes parameter = exp(-gamma * r)
                for j in range(0,n):
                    r = np.sqrt(np.sum((ns[i,:]-ns[j,:])**2))

                    if light[i] > lightO[j]:
                        beta0=1
                        beta=(beta0-betaMin)*math.exp(-gamma*r**2)+betaMin
                        tmpf=alpha*(np.random.rand(dim)-0.5)*scale
                        ns[i,:]=ns[i,:]*(1-beta)+nso[j,:]*beta+tmpf
                        #sometimes it happes to be 0. Better save then sorry
                        ns[i][ns[i] == 0] = 1


            iterationNumber = k
            bestQuality = fbest

            print "Iteration number:"
            print k
            print "Best value function:"
            print fbest
            print "for vector:"
            print nbest

    def calcValue(self,m):
        """
        """
        avrTimes = self.bcmp_interface(m.tolist())
        timeSum = 0
        for i in avrTimes:
            for j in i:
                timeSum += j

        if timeSum < 0:
            timeSum = 0

        functionValue = 0

        for i in m:
            functionValue += i*0.01
        functionValue += timeSum * 1000

        return functionValue

    def bcmp_interface(self, m):
        types = self.bcmp_config.get('types')
        nr_of_class_one = sum(i == 1 for i in types)
        if nr_of_class_one != len(m):
            raise ValueError("Amount of ms must be equal to nodes with type = 1")

        final_m = [1 if type > 1 else m.pop(0) for type in types]

        solver = BcmpNetworkClosed(
            R=self.bcmp_config['R'],
            N=self.bcmp_config['N'],
            k=self.bcmp_config['k'],
            mi_matrix=self.bcmp_config['mi'],
            p=self.bcmp_config['classes'],
            node_info=zip(self.bcmp_config['types'], final_m),
            epsilon=self.bcmp_config['epsilon']
        )
        vals = solver.get_measures()

        return vals['mean_w_matrix']

    def bcmpParams(self):
        """
        Initial params of BCMP queue network
        Optinally read it from file
        """
        # Classes amount
        R = 2
        # Nodes amount
        N = 3
      
        # service times
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
        # As there's no class transitions we can safely use list of
        # single class transition probabilities
        classes = [p1, p2]

        # generate network parameters
        # node types
        types = [1, 1, 1]
        # servers amount in respect to node type
        #m = [3, 1, 1]
        # amount of request by class
        K = [200,100]
        epsilon = 0.0001
        return R, N, K, mi, classes, types, epsilon


if __name__ == "__main__":
    m = [6, 4, 4]
    solution = FireflyAlgorithm()

    solution.FFA(**solution.ffa_config)
    print solution.bcmp_interface(m)

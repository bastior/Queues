import numpy as np
import random
import math
from  bcmp_closed import *



class FireflyAlgorythm(object):
    def __init__(object):
        pass

    def FFA(self, objectiveFunction, lowerBound, upperBound, dim, n, maxGeneration):
        """
        objectiveFunction - function that calcs value for each firefly (vector) of solution
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

        light = np.ones(n)
        light.fill(float("inf"))

        #main loop
        for k in range(0,maxGeneration):

            for i in range(0,n):
                #LIZONCZYK TUTAJ WRZUCIC zn[i] = objectiveFunction(ns) gdzie ns jest wektorem
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


            iterationNumber = k
            bestQuality = fbest

            print "Iteration number:"
            print k
            print "Best value function:"
            print fbest
            print "for vector:"
            print nbest
            

    def calcValue(m):
        """
        Wymyslec i policzyc funkcje celu na podstawie M i srednich czasow oczekiwania
        """
        pass

    def bcmpIf(self, m):
        params = self.bcmpParams()
        nrOfClassOne = sum(i == 1 for i in params[5])
        if (nrOfClassOne != len(m)):
            raise ValueError("Amount of ms must be equal to nodes with type = 1")

        indices = [i for i,j in enumerate(params[5]) if j == 1]
        finalM = [0] * (len(params[5]))
        for i in range(0,len(params[5])):
            if (params[5][i] > 1):
                finalM[i] = 1
            else:
                finalM[i] = m.pop(0)

        solver = BcmpNetworkClosed(
            R=params[0],
            N=params[1],
            k=params[2],
            mi_matrix=params[3],
            p=params[4],
            node_info=zip(params[5], finalM),
            epsilon=params[6]
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
        types = [1, 1, 3]
        # servers amount in respect to node type
        #m = [3, 1, 1]
        # amount of request by class
        K = [3, 3]
        epsilon = 0.0001
        return R, N, K, mi, classes, types, epsilon

if __name__ == "__main__":
    #m = [2,1]
    solution = FireflyAlgorythm()

    #solution.FFA(1, 1, 2, 2, 2, 2)
    #solution.bcmpIf(m)
    pass

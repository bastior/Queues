import numpy as np
import random
import math
from  bcmp_closed import *



class FireflyAlgorythm(object):
    def __init__(object):
        pass

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
        avrTimes = self.bcmpIf(m.tolist())
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
        mi = np.matrix([[1, 4],
                        [2, 5],
                        [3, 0.99]])
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
    m = [7,2,3]
    solution = FireflyAlgorythm()

    #solution.FFA(2, 10, 3, 15, 10)
    print solution.bcmpIf(m)
    pass

import numpy as np


class NumpyMath():

    def addf(self, a, b):
        return np.add(np.array(a, dtype=float), np.array(b, dtype=float), dtype=float)

    def addi(self, a, b):
        return np.add(np.array(a, dtype=int), np.array(b, dtype=int), dtype=int)

    def subtractf(self, a, b):
        return np.subtract(np.array(a, dtype=float), np.array(b, dtype=float), dtype=float)

    def subtracti(self, a, b):
        return np.subtract(np.array(a, dtype=int), np.array(b, dtype=int), dtype=int)
        
    def multiplyf(self, a, b):
        return np.multiply(np.array(a, dtype=float), np.array(b, dtype=float), dtype=float)

    def multiplyi(self, a, b):
        return np.multiply(np.array(a, dtype=int), np.array(b, dtype=int), dtype=int)
        
    def dividef(self, a, b):
        return np.divide(np.array(a, dtype=float), np.array(b, dtype=float), dtype=float)

    def dividei(self, a, b):
        return np.divide(np.array(a, dtype=int), np.array(b, dtype=int), dtype=int)
        
    def sumf(self, a):
        return np.sum(np.array(a, dtype=float), dtype=float)

    def sumi(self, a):
        return np.sum(np.array(a, dtype=int), dtype=int)

    def minf(self, a):
        return np.min(np.array(a, dtype=float), dtype=float)

    def mini(self, a):
        return np.max(np.array(a, dtype=int), dtype=int)
    
    def maxf(self, a):
        return np.max(np.array(a, dtype=float), dtype=float )

    def maxi(self, a):
        return np.max(np.array(a, dtype=int), dtype=int)

    def logf(self, a):
        return np.log(np.array(a, dtype=float), dtype=float)

    def logi(self, a):
        return np.log(np.array(a, dtype=int), dtype=int)





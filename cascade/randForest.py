import math
import numpy

class RandForest(object):
    def __init__(self, paras):
        self.treeNum   = paras["treeNum"]
        self.treeDepth = paras["treeDepth"]
        self.feaNum    = paras["feaNum"]
        self.radius    = paras["radius"]
        self.tree      = []

    def train(self, data):
        ###Step1 : Generate Feature 
        for n in xrange(self.treeNum):
            tree = RegTree()

    def genBinaryFea(self, data):
        pass


class RegTree(object):
    def __init__(self, paras):
        self.depth  = paras["depth"]
        self.radius = paras["radius"]

    def train(self, fea, x, y):
        """
        fea : [num, dim]
        """
        
        ### Gen feature first
        
        feaDim
        pass

    def genFeaType(self, num):
        feaTypes = numpy.zeros((num, 4))
        radRange, angRange = 30, 36
        a = numpy.array(range(0, (radRange+1)*(angRange+1)))
        b = numpy.array(range(0, (radRange+1)*(angRange+1)))
        random.shuffle(a)
        random.shuffle(b)
        dif_idx = a!=b
        a=a[dif_idx]
        b=b[dif_idx]
        a=a[0:num]
        b=b[0:num]
        
        for i in range(num):
            rad_a = floor(a[i]/angRange)
            ang_a = floor(a[i]%angRange)
            rad_b = floor(b[i]/angRange)
            ang_b = floor(b[i]%angRange)
            feaType[i, :] = (rad_a/radRange*self.radius,
                             ang_a/angRange*2*math.pi,
                             rad_b/radRange*self.radius,
                             ang_b/angRange*2*math.pi) 
        return feaType
    

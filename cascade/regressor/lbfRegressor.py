import sys
import os
import numpy
import time
from utils   import *
from randForest import *

class LBFRegressor(object):
    """
    Face Alignment at 3000 FPS via Regressing LBF
    """
    def __init__(self, paras):     
        self.maxTreeNum = paras["maxTreeNum"]
        self.treeDepth  = paras["treeDepth"]
        self.feaNum     = paras["feaNum"]
        self.radius     = paras["radius"]
        self.rfs = []

    def train(self, trainSet):
        pntNum = trainSet.meanShape.shape[0]
        treeNum = int(self.maxTreeNum/pntNum)
        
        for i in xrange(pntNum):
            rf = RandForest(treeDepth = self.treeDepth,
                            treeNum   = treeNum,
                            feaNum    = self.feaNum,
                            radius    = self.radius)
            
            rf.train(trainSet, i)

            self.rfs.append(rf)

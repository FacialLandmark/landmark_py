import sys
import os
import numpy as NP
from scipy.sparse import lil_matrix
from sklearn.svm import LinearSVR
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
        self.binNum     = paras["binNum"]
        self.feaRange   = paras["feaRange"]
        self.rfs = []
        self.regs = []

    def train(self, trainSet):
        pntNum = trainSet.meanShape.shape[0]
        treeNum = int(self.maxTreeNum/pntNum)
        
        ### Train the random forests
        begTime = time.time()
        for i in xrange(pntNum):
            rf = RandForest(treeDepth = self.treeDepth,
                            treeNum   = treeNum,
                            feaNum    = self.feaNum,
                            radius    = self.radius,
                            binNum    = self.binNum,
                            feaRange  = self.feaRange)
            rf.train(trainSet, i)
            self.rfs.append(rf)
        elapse = getTimeByStamp(begTime, time.time(), 'min')
        print("\t\tRandom Forest     : %f mins"%elapse)

        ### Extract the local binary features
        begTime = time.time()
        feas = self.genFeaOnTrainset(trainSet)
        elapse = getTimeByStamp(begTime, time.time(), 'min')
        print("\t\tExtract LBFs      : %f mins"%elapse)

        ### Global regression
        begTime = time.time()
        y = trainSet.residuals
        y = y.reshape(y.shape[0], y.shape[1]*y.shape[2])
        for i in xrange(pntNum*2):
            ### TODO Show the training result 
            reg=LinearSVR(epsilon=0.0, 
                          C = 1.0/feas.shape[0],
                          loss='squared_epsilon_insensitive',
                          fit_intercept = True)
            reg.fit(feas, y[:, i])
            self.regs.append(reg)
        elapse = getTimeByStamp(begTime, time.time(), 'min')
        print("\t\tGlobal Regression : %f mins"%elapse)

        ### Update the initshapes
        begTime = time.time()
        for i in xrange(pntNum):
            regX = self.regs[2*i]
            regY = self.regs[2*i+1]
            
            x = regX.predict(feas)
            y = regY.predict(feas)
            delta = NP.squeeze(NP.dstack((x,y)))
            delta = Affine.transPntsForwardWithDiffT(delta, 
                                                     trainSet.ms2reals)
            delta = NP.multiply(delta, 
                                trainSet.bndBoxs[:,[2,3]])
            trainSet.initShapes[:,i,:] = trainSet.initShapes[:,i,:] + delta
        elapse = getTimeByStamp(begTime, time.time(), 'min')
        print("\t\tUpdate Shape      : %f mins"%elapse)
            
    def detect(self, img, bndbox, initShape, affineT):
        ### Extract features
        fea = self.extractFea(img, bndbox, 
                              initShape, affineT)
        pntNum = initShape.shape[0]
        ### Get the residules
        for i in xrange(pntNum):
            regX = self.regs[2*i]
            regY = self.regs[2*i+1]
            
            x = regX.predict(fea)
            y = regY.predict(fea)
            delta = NP.squeeze(NP.dstack((x,y)))
            delta = Affine.transPntForward(delta, affineT)
            delta = NP.multiply(delta, (bndbox[2],bndbox[3]))
            initShape[i,:] = initShape[i,:] + delta

    def getFeaDim(self):
        feaDim = 0
        for rf in self.rfs:
            for tree in rf.trees:
                feaDim = feaDim + tree.leafNum
        return feaDim


    def extractFea(self, img, bndbox, initShape, affineT):
        feaDim = self.getFeaDim()
        fea = lil_matrix((1, feaDim), 
                         dtype=NP.int8)

        offset = 0
        for j, rf in enumerate(self.rfs):
            point = initShape[j]
            for t in rf.trees:
                ### TODO judge the empty tree
                leafIdx, dim = t.genBinaryFea(img, 
                                              bndbox, 
                                              affineT, 
                                              point)
                fea[0, offset+leafIdx] = 1
                offset = offset + dim                    
        return fea

    def genFeaOnTrainset(self, trainSet):
        feaDim = self.getFeaDim()
        sampleNum = trainSet.initShapes.shape[0]
        feas = lil_matrix((sampleNum, feaDim), 
                          dtype=NP.int8)
        
        augNum = trainSet.augNum
        for i in xrange(sampleNum):
            imgData = trainSet.imgDatas[i/augNum] 
            bndBox  = trainSet.bndBoxs[i] 
            affineT = trainSet.ms2reals[i]
            shape   = trainSet.initShapes[i]
            
            offset = 0
            for j, rf in enumerate(self.rfs):
                point = shape[j]
                for t in rf.trees:
                    ### TODO judge the empty tree
                    leafIdx, dim = t.genBinaryFea(imgData, 
                                                  bndBox, 
                                                  affineT, 
                                                  point)
                    feas[i, offset+leafIdx] = 1
                    offset = offset + dim                    
        return feas
        

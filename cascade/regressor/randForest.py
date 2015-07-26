import math
import numpy as NP
import random
from dator import *

class RandForest(object):
    def __init__(self, 
                 treeDepth = None, 
                 treeNum   = None,
                 feaNum    = None, 
                 radius    = None,
                 binNum    = None,
                 feaRange  = None):
        self.treeDepth = treeDepth
        self.treeNum   = treeNum
        self.feaNum    = feaNum
        self.radius    = radius
        self.binNum    = binNum
        self.feaRange  = feaRange
        self.trees     = []

    def train(self, train_set, pointIdx):    
        sampleNum = train_set.initShapes.shape[0]
        
        for n in xrange(self.treeNum):
            ### Get train example indexs by bootstrape
            sampleIdxs = self.bootStrape(sampleNum)
            
            ### Construct tree to train
            tree = RegTree(depth  = self.treeDepth, 
                           radius = self.radius,
                           feaNum = self.feaNum,
                           binNum = self.binNum,
                           feaRange = self.feaRange)
            tree.train(train_set, pointIdx, sampleIdxs)
            self.trees.append(tree)
            
    def bootStrape(self, sampleNum):
        """
        Todo : Try different bootstrape method, 
        Now Follow the method from matlab
        """        
        ### 
        overlap = 0.4
        subNum = int(sampleNum/((1-overlap )*self.treeNum))
        if not hasattr(self, 'treeIdx'):
            self.treeIdx = 0

        beg = self.treeIdx*subNum*(1-overlap)
        beg = max(0, int(beg))
        end = min(beg+subNum, sampleNum-1)      
        self.treeIdx += 1        
        return NP.array(range(beg, end+1))


class RegTree(object):
    def __init__(self, 
                 depth    = None, 
                 radius   = None, 
                 feaNum   = None,
                 binNum   = None,
                 feaRange = None):
        ### paras
        self.depth  = depth
        self.radius = radius
        self.feaNum = feaNum
        self.binNum = binNum
        self.feaRange = feaRange

        ### tree 
        self.leafNum = 0
        self.tree    = None
        
    def genBinaryFea(self, imgData, bndBox, affineT, point):
        tree = self.tree
        imgH, imgW  = imgData.shape
        w, h = bndBox[2:4]
        point_a = NP.zeros(2, dtype=point.dtype)
        point_b = NP.zeros(2, dtype=point.dtype)
        while 'leafIdx' not in tree:
            feaType  = tree["feaType"] 
            feaRange = tree["feaRange"]
            th = tree["threshold"]
            
            angle_cos = NP.cos(feaType[[1,3]])
            angle_sin = NP.sin(feaType[[1,3]])
            ms_x_ratio = angle_cos*feaType[[0,2]]
            ms_y_ratio = angle_sin*feaType[[0,2]]
            
            point_a[0] = ms_x_ratio[0]*w
            point_a[1] = ms_y_ratio[0]*h
            point_b[0] = ms_x_ratio[1]*w
            point_b[1] = ms_y_ratio[1]*h
            
            ### convert meanshape coord into real coord
            point_a = Affine.transPntForward(point_a,
                                             affineT)
            point_b = Affine.transPntForward(point_b, 
                                             affineT)
            point_a = point_a + point
            point_b = point_b + point
            
            ### TODO use other interpolations
            point_a = NP.around(point_a)
            point_b = NP.around(point_b)
            
            ### Check with the image size
            point_a[point_a<0]=0             
            point_b[point_b<0]=0 
            if point_a[0]>imgW-1:
                point_a[0]=imgW-1
            if point_a[1]>imgH-1:
                point_a[1]=imgH-1
            if point_b[0]>imgW-1:
                point_b[0]=imgW-1
            if point_b[1]>imgH-1:
                point_b[1]=imgH-1

            ### Construct the idx list for get the elements
            fea = NP.subtract(imgData[point_a[1], 
                                      point_a[0]] ,
                              imgData[point_b[1], 
                                      point_b[0]],
                              dtype=NP.float32)

            ### get the diff          
            fea = (fea-feaRange[0])/feaRange[2]
            if fea <= th:
                tree = tree["left"]
            else:
                tree = tree["right"]
        
        leafIdx = tree["leafIdx"]
        return leafIdx, self.leafNum        
                
    def train(self, train_set, pointIdx, sampleIdxs):
        self.tree = self.split(train_set, 
                               pointIdx,
                               sampleIdxs)

    def split(self, train_set, pointIdx, sampleIdxs):        
        tree = {}
        if self.depth<0 or len(sampleIdxs)<2:
            tree["leafIdx"] = self.leafNum
            self.leafNum = self.leafNum+1
            return tree

        ### Get the current residuals
        errs = train_set.residuals[sampleIdxs, pointIdx]
        
        ### Generate feature types     
        feaTypes = self.genFeaType(self.feaNum)

        ### Extract the pixel difference feature
        pdFeas = self.genFea(train_set, pointIdx, 
                             sampleIdxs,feaTypes)        

        ### Normalize the feature
        minFeas, maxFeas, feaSteps = self.normalize(pdFeas) 
        
        ### Find the best feature and threshold      
        bestIdx, th = self.findBestSplit(pdFeas, errs)

        ### split left and right leaf recurrently
        lIdx = pdFeas[:, bestIdx]<=th
        rIdx = pdFeas[:, bestIdx]>th
        lSamples = sampleIdxs[lIdx]
        rSamples = sampleIdxs[rIdx]
        self.depth = self.depth - 1
        tree["feaType"] = feaTypes[bestIdx]
        tree["feaRange"] = (minFeas[bestIdx], 
                            maxFeas[bestIdx],
                            feaSteps[bestIdx])
        tree["threshold"] = th
        tree["left"] = self.split(train_set, 
                                  pointIdx, 
                                  lSamples)
        tree["right"] = self.split(train_set, 
                                   pointIdx, 
                                   rSamples)   
        return tree
        
    def findBestSplit(self, feas, errs):
        sampNum, feaNum = feas.shape
        sortedFeas = NP.sort(feas, axis=0)        
        lossAndTh = NP.zeros((feaNum, 2))
        
        for idxFea in xrange(feaNum):      
            ### Randomly split on each feature              
            ### TODO choose the best split
            ind = int(sampNum*(0.5 + 0.9*(random.random()-0.5)));
            th = sortedFeas[ind, idxFea]
            lIdx = feas[:, idxFea]<=th
            rIdx = feas[:, idxFea]>th
                                
            lErrs = errs[lIdx]
            rErrs = errs[rIdx]
            lNum  = lErrs.shape[0]
            rNum  = rErrs.shape[0]   
            if lNum<2:
                lVar = 0;
            else:
                lVar = NP.sum(NP.mean(NP.power(lErrs, 2), 
                                      axis=0) - 
                              NP.power(NP.mean(lErrs, 
                                               axis=0),2))
            if rNum < 2:
                rVar = 0
            else:
                rVar = NP.sum(NP.mean(NP.power(rErrs, 2), 
                                      axis=0) - 
                              NP.power(NP.mean(rErrs, 
                                               axis=0),2))
            lossAndTh[idxFea] = (lNum*lVar + rNum*rVar, th) 
        bestFeaIdx = lossAndTh[:,0].argmin()
        return bestFeaIdx, lossAndTh[bestFeaIdx, 1]
        
        
    def genFea(self, train_set, 
               pointIdx, sampleIdxs, feaTypes):
        sampleNum = len(sampleIdxs)        
        feaNum = feaTypes.shape[0]
        pdFea = NP.zeros((sampleNum, feaNum), 
                         dtype=NP.float32)

        coord_a = NP.zeros((feaNum, 2))
        coord_b = NP.zeros((feaNum, 2))
        angle_cos = NP.cos(feaTypes[:, [1,3]])
        angle_sin = NP.sin(feaTypes[:, [1,3]])

        ms_x_ratio = angle_cos*feaTypes[:, [0,2]]
        ms_y_ratio = angle_sin*feaTypes[:, [0,2]]

        augNum = train_set.augNum
        for i, idx in enumerate(sampleIdxs):              
            T = train_set.ms2reals[idx]
            bndBox = train_set.bndBoxs[idx]
            imgData = train_set.imgDatas[idx/augNum]        
            initShape = train_set.initShapes[idx]
            
            imgH, imgW  = imgData.shape
            w, h = bndBox[2:4]
            coord_a[:, 0] = ms_x_ratio[:, 0]*w
            coord_a[:, 1] = ms_y_ratio[:, 0]*h
            coord_b[:, 0] = ms_x_ratio[:, 1]*w
            coord_b[:, 1] = ms_y_ratio[:, 1]*h

            ### convert meanshape coord into real coord
            coord_a = Affine.transPntsForwardWithSameT(coord_a, T)
            coord_b = Affine.transPntsForwardWithSameT(coord_b, T)
            coord_a = coord_a+initShape[pointIdx]
            coord_b = coord_b+initShape[pointIdx]
            
            ### TODO use other interpolations
            coord_a = NP.around(coord_a)
            coord_b = NP.around(coord_b)
            
            ### Check with the image size
            coord_a[coord_a<0]=0 
            coord_a[coord_a[:,0]>imgW-1, 0]=imgW-1 
            coord_a[coord_a[:,1]>imgH-1, 1]=imgH-1 
            coord_b[coord_b<0]=0 
            coord_b[coord_b[:,0]>imgW-1, 0]=imgW-1 
            coord_b[coord_b[:,1]>imgH-1, 1]=imgH-1 

            ### Construct the idx list for get the elements
            idx_a = NP.transpose(coord_a).tolist()
            idx_a[0], idx_a[1] = idx_a[1], idx_a[0]
            idx_b = NP.transpose(coord_b).tolist()
            idx_b[0], idx_b[1] = idx_b[1], idx_b[0]

            ### get the diff          
            pdFea[i,:] = NP.subtract(imgData[idx_a],
                                     imgData[idx_b],
                                     dtype = NP.int16)
        return pdFea

    def genFeaType(self, num):
        feaType = NP.zeros((num, 4), dtype=NP.float32)
        radRange, angRange = 30, 36
        a = NP.array(range(0, (radRange+1)*(angRange+1)),
                        dtype=NP.float32)
        b = NP.array(range(0, (radRange+1)*(angRange+1)),
                        dtype=NP.float32)
        random.shuffle(a)
        random.shuffle(b)
        dif_idx = a!=b
        a=a[dif_idx]
        b=b[dif_idx]
        a=a[0:num]
        b=b[0:num]
        
        for i in range(num):
            rad_a = math.floor(a[i]/(angRange+1))
            ang_a = math.floor(a[i]%(angRange+1))
            rad_b = math.floor(b[i]/(angRange+1))
            ang_b = math.floor(b[i]%(angRange+1))
            feaType[i, :] = (rad_a/radRange*self.radius,
                             ang_a/angRange*2*math.pi,
                             rad_b/radRange*self.radius,
                             ang_b/angRange*2*math.pi) 
        return feaType
    
    def normalize(self, feas):
        feaDim = feas.shape[1]        
        minFeas = NP.empty(feaDim, 
                           dtype = NP.float32)
        maxFeas = NP.empty(feaDim, 
                           dtype = NP.float32)
        feaSteps= NP.empty(feaDim, 
                           dtype = NP.float32)

        if None != self.feaRange:
            minFeas[:] = self.feaRange[0]
            maxFeas[:] = self.feaRange[1]
        else:
            NP.min(feas, axis=0, out=minFeas)
            NP.max(feas, axis=0, out=maxFeas)
        feaR = (maxFeas - minFeas + 1)
        feaSteps[:] = feaR/self.binNum
        NP.subtract(feas, minFeas, out=feas)
        NP.divide(feas, feaSteps, out=feas, dtype=NP.float32)
        NP.round(feas, out=feas)
        return minFeas, maxFeas, feaSteps

import math
import numpy
import random
from ..dator import *
class RandForest(object):
    def __init__(self, treeDepth=None, treeNum=None,
                 feaNum=None, radius=None):
        self.treeDepth = treeDepth
        self.treeNum   = treeNum
        self.feaNum    = feaNum
        self.radius    = radius
        self.trees     = []

    def train(self, train_set, pointIdx):    
        sampleNum = train_set.initShapes.shape[0]
        
        for n in xrange(self.treeNum):
            ### Get train example indexs by bootstrape
            sampleIdxs = self.bootStrape(sampleNum)
            
            ### Construct tree to train
            tree = RegTree(depth  = self.treeDepth, 
                           radius = self.radius,
                           feaNum = self.feaNum)
            tree.train(train_set, pointIdx, sampleIdxs)
            self.trees.append(tree)
            
    def bootStrape(self, sampleNum):
        """
        Todo : Try different bootstrape method
        """
        
        ### Follow the method from matlab
        overlap = 0.4
        subNum = int(sampleNum/((1-overlap )*self.treeNum))
        if not hasattr(self, 'treeIdx'):
            self.treeIdx = 0

        beg = self.treeIdx*subNum*(1-overlap)
        beg = max(0, int(beg))
        end = min(beg+subNum, sampleNum-1)      
        self.treeIdx += 1
        
        return range(beg, end+1)


class RegTree(object):
    def __init__(self, depth=None, radius=None, feaNum=None):
        self.depth  = depth
        self.radius = radius
        self.feaNum = feaNum

    def train(self, train_set, pointIdx, sampleIdxs):
        """
        fea : [num, dim]
        """        
        return self.split(train_set, 
                          pointIdx,
                          sampleIdxs, self.depth)

    def split(self, train_set, pointIdx, sampleIdxs, depth):
        depth -= 1
        if depth < 0 or len(sampleIdxs)<2:
            return 
        
        ### Generate feature types     
        feaTypes = self.genFeaType(self.feaNum)

        ### Generate the pixel differences
        pdFea = self.genFea(train_set, 
                            pointIdx, 
                            sampleIdxs, 
                            feaTypes)        
        
        ### split left and right leaf recurrently
        pass

    def genFea(self, train_set, 
               pointIdx, sampleIdxs, feaTypes):
        sampleNum = len(sampleIdxs)        
        feaNum = feaTypes.shape[0]
        pdFea = numpy.zeros((sampleNum, feaNum))

        coord_a = numpy.zeros((feaNum, 2))
        coord_b = numpy.zeros((feaNum, 2))
        angle_cos = numpy.cos(feaTypes[:, [1,3]])
        angle_sin = numpy.sin(feaTypes[:, [1,3]])

        ms_x_ratio = angle_cos*feaTypes[:, [0,2]]
        ms_y_ratio = angle_sin*feaTypes[:, [0,2]]

        for i, idx in enumerate(sampleIdxs):              
            T = train_set.ms2reals[idx]
            bndBox = train_set.bndBoxs[idx]
            imgData = train_set.imgDatas[idx]            
            initShape = train_set.initShapes[idx]
            
            imgH, imgW  = imgData.shape
            w, h = bndBox[2:4]
            coord_a[:, 0] = ms_x_ratio[:, 0]*w
            coord_a[:, 1] = ms_y_ratio[:, 0]*h
            coord_b[:, 0] = ms_x_ratio[:, 1]*w
            coord_b[:, 1] = ms_y_ratio[:, 1]*h

            ### convert meanshape coord into real coord
            coord_a = Affine.transPointsForward(coord_a, T)
            coord_b = Affine.transPointsForward(coord_b, T)
            coord_a = coord_a+initShape[pointIdx]
            coord_b = coord_b+initShape[pointIdx]
            
            ### TODO use other interpolations
            coord_a = coord_a.astype(numpy.int32)
            coord_b = coord_b.astype(numpy.int32)
            
            ### Check with the image size
            coord_a[coord_a<0]=0 
            coord_a[coord_a[:,0]>imgW-1, 0]=imgW-1 
            coord_a[coord_a[:,1]>imgH-1, 0]=imgH-1 
            coord_b[coord_b<0]=0 
            coord_b[coord_b[:,0]>imgW-1, 0]=imgW-1 
            coord_b[coord_b[:,1]>imgH-1, 0]=imgH-1 

            ### Construct the idx list for get the elements
            idx_a = numpy.transpose(coord_a).tolist()
            idx_a[0], idx_a[1] = idx_a[1], idx_a[0]
            idx_b = numpy.transpose(coord_b).tolist()
            idx_b[0], idx_b[1] = idx_b[1], idx_b[0]

            ### get the diff          
            pdFea[i,:] = imgData[idx_a] - imgData[idx_b]
        print pdFea

    def genFeaType(self, num):
        feaType = numpy.zeros((num, 4))
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
            rad_a = math.floor(a[i]/angRange)
            ang_a = math.floor(a[i]%angRange)
            rad_b = math.floor(b[i]/angRange)
            ang_b = math.floor(b[i]%angRange)
            feaType[i, :] = (rad_a/radRange*self.radius,
                             ang_a/angRange*2*math.pi,
                             rad_b/radRange*self.radius,
                             ang_b/angRange*2*math.pi) 
        return feaType
    

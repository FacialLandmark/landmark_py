import os
import numpy as np
import copy
import math

from utils  import *       
from affine import *
from shape  import *
from reader import *

class TrainSet(object):
    def __init__(self):
        self.imgDatas = []
        self.gtShapes = []
        self.bndBoxs  = []        
        self.initShapes = []
        self.ms2reals  = []
        self.real2mss  = []
        self.meanShape = None
        self.augNum = 1

    def add(self, img, gtShape, bndBox):
        self.imgDatas.append(img)
        self.gtShapes.append(gtShape)
        self.bndBoxs.append(bndBox)
    
    def calMeanShape(self):              
        meanShape = np.zeros(self.gtShapes[0].shape)
        for i, s in enumerate(self.gtShapes):
            normS = Shape.shapeReal2Norm(s, self.bndBoxs[i])
            meanShape = np.add(meanShape, normS)
            
        self.meanShape = meanShape/len(self.gtShapes)

    def genTrainData(self, augNum):
        ### Step1 : Compute the mean shape
        self.calMeanShape()
        
        ### Set meanshape as the initshape
        for bb in self.bndBoxs:
            initShape = Shape.shapeNorm2Real(self.meanShape,
                                             bb)
            self.initShapes.append(initShape)
            
        ### Translate list into numpy's array
        self.initShapes = np.asarray(self.initShapes,
                                     dtype = np.float32)
        self.gtShapes   = np.asarray(self.gtShapes,
                                     dtype = np.float32)
        self.bndBoxs    = np.asarray(self.bndBoxs,
                                     dtype = np.float32)
        
        ### Shape augment            
        if augNum > 1:
            self.augNum = augNum
            self.initShapes = np.repeat(self.initShapes,
                                        augNum,
                                        axis = 0)
            self.gtShapes = np.repeat(self.gtShapes,
                                      augNum,
                                      axis = 0)
            self.bndBoxs = np.repeat(self.bndBoxs,
                                     augNum,
                                     axis = 0)
            ### Permutate the augmented shape
            sampleNum = self.initShapes.shape[0]
            for i in xrange(sampleNum):
                if 0==i%sampleNum:
                    continue
                shape = self.initShapes[i]
                self.initShapes[i]= Shape.augment(shape)
        return
    
    def getAffineT(self):
        num = self.gtShapes.shape[0]
        self.ms2real  = []
        self.real2ms  = []

        for i in range(num):
            ### Project to meanshape coordinary    
            bndBox = self.bndBoxs[i]
            initShape = self.initShapes[i]
            mShape = Shape.shapeNorm2Real(self.meanShape,
                                         bndBox)             
            T = Affine.fitGeoTrans(initShape, mShape)   
            self.real2mss.append(T)
            T = Affine.fitGeoTrans(mShape, initShape)
            self.ms2reals.append(T)

    def calResiduals(self):       
        ### Compute the affine matrix 
        self.getAffineT()
        
        self.residuals = np.zeros(self.gtShapes.shape)
        num = self.gtShapes.shape[0]
        for i in range(num):
            ### Project to meanshape coordinary              
            T = self.real2mss[i]
            bndBox = self.bndBoxs[i]
            err = self.gtShapes[i]-self.initShapes[i]
            err = np.divide(err, (bndBox[2], bndBox[3]))
            err = Affine.transPntsForwardWithSameT(err, T)
            self.residuals[i,:] = err
    
        
class DataWrapper(object):
    def __init__(self, para): 
        self.path = para['path']
        self.augNum = para['augNum']

        if 'dataset' in para:
            if 'aflw' == para['dataset'].lower():
                self.reader = AFLWReader
        else:
            self.reader = SelfReader

    def read(self):
        if not os.path.exists(self.path):
            raise Exception("Train set not exist")     
        trainSet = TrainSet()
       
        paths = open(self.path).readlines()
        for imgP in paths:
            try:                
                img, gtShape = self.reader.read(imgP)
                ### Crop the image
                bndBox = Shape.getBBoxByPts(gtShape)
                cropB, img = self.cropRegion(bndBox, 2, img)
                gtShape = np.subtract(gtShape, 
                                      (cropB[0], cropB[1]))

                ### TODO Add rotation to augment the sample
                ### TODO Use face detector to detect the face
                ### Get the bndBox.
                bndBox = Shape.getBBoxByPts(gtShape)
                trainSet.add(img, gtShape, bndBox)
            except:
                pass
        
        ### Generate the meanShape
        trainSet.genTrainData(self.augNum)
        return trainSet

    def cropRegion(self, bbox, scale, img):
        height, width = img.shape
        w = math.floor(scale*bbox[2])
        h = math.floor(scale*bbox[3])
        x = max(0, math.floor(bbox[0]-(w-bbox[2])/2))
        y = max(0, math.floor(bbox[1]-(h-bbox[3])/2))
        w = min(width-x, w)
        h = min(height-y, h)     
        
        ### If not use deepcopy, the subImg will hold the whole img's memory
        subImg = copy.deepcopy(img[y:y+h, x:x+w])
        return (x,y,w,h), subImg
    
    def printParas(self):
        print('\tDataset     = %s'%(self.path))
        print('\tAugment Num = %d'%(self.augNum))


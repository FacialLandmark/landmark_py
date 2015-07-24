import os
import sys
import numpy as np
from   numpy import loadtxt
from PIL import Image
from utils import *       
from affine import *
import copy
import math

### API for augment data
class Shape(object):
    @classmethod
    def shapeReal2Norm(cls, realShape, bndBox):
        normShape = np.subtract(realShape, 
                                (bndBox[0],bndBox[1]))
        normShape = np.divide(normShape, 
                              (bndBox[2]-1,bndBox[3]-1))
        return normShape
    
    @classmethod
    def shapeNorm2Real(cls, normShape, bndBox):
        realShape = np.multiply(normShape, 
                                (bndBox[2]-1, bndBox[3]-1))
        realShape = np.add(realShape, 
                           (bndBox[0],bndBox[1]))
        return realShape
    
    def augment(cls, shape):
        shape = cls.scale(shape)
        shape = cls.rotate(shape)
        shape = cls.shift(shape)
        return shape

    def scale(cls, shape):
        scale = 1 + 0.2*(RD.random()-0.5)
        cent = np.mean(shape, axis=0)
        newShape = scale*(shape-cent) + cent
        return newShape

    def rotate(cls, shape):
        return shape

    def shift(cls, shape):
        return shape

class TrainSet(object):
    def __init__(self):
        self.imgDatas = []
        self.gtShapes = []
        self.bndBoxs  = []        
        self.initShapes = []
        self.ms2reals  = []
        self.real2mss  = []
        self.meanShape = None

    def add(self, img, gtShape, bndBox):
        self.imgDatas.append(img)
        self.gtShapes.append(gtShape)
        self.bndBoxs.append(bndBox)
    
    def calMeanShape(self):              
        meanShape = np.zeros(self.gtShapes[0].shape)
        for idx, s in enumerate(self.gtShapes):
            shape = Shape.shapeReal2Norm(s, 
                                         self.bndBoxs[idx])
            meanShape = np.add(meanShape, shape)
            
        self.meanShape = meanShape/len(self.gtShapes)

    def genTrainData(self, augNum):
        ### Step1 : Compute the mean shape
        self.calMeanShape()
        
        ### Set meanshape as the initshape
        for bb in self.bndBoxs:
            initShape = Shape.shapeNorm2Real(self.meanShape,
                                             bb)
            self.initShapes.append(initShape)

        ### Todo : Add augment            
            
        ### Trans list into numpy's array
        self.initShapes = np.asarray(self.initShapes,
                                     dtype = np.float32)
        self.gtShapes   = np.asarray(self.gtShapes,
                                     dtype = np.float32)
        self.bndBoxs    = np.asarray(self.bndBoxs,
                                     dtype = np.float32)
        ### Todo : shuffle the train set 
    
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
                bndBox = self.getBBoxByPts(gtShape)
                cropB, img = self.cropRegion(bndBox, 2, img)
                gtShape = np.subtract(gtShape, 
                                      (cropB[0], cropB[1]))

                ### TODO 1. add rotation to augment the sample
                ### TODO 2. Use face detector to detect the face
                ### Get the bndBox.
                bndBox = self.getBBoxByPts(gtShape)
                trainSet.add(img, gtShape, bndBox)
            except:
                pass
        
        ### Generate the meanShape
        trainSet.genTrainData(self.augNum)
        return trainSet

    def getBBoxByPts(self, pts):
        maxV = np.max(pts, axis=0)
        minV = np.min(pts, axis=0)
        return (minV[0], minV[1],
                maxV[0]-minV[0]+1,
                maxV[1]-minV[1]+1)    

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

    
class SelfReader(object):
    """
    self contained
    """
    @classmethod
    def read(cls, imgPath):
        imgP = imgPath.strip()
        folder, name = os.path.split(imgP)
        file_name,_ = os.path.splitext(name)
        folder, id_name = os.path.split(folder)
        annP = "%s/Annotations/%s/%s_face.txt"%(folder,
                                                id_name,
                                                file_name)
        
        ### Load the ground truth of shape
        gtShape = loadtxt(annP, comments="#", 
                          delimiter=",",
                          unpack=False)
        gtShape = gtShape.astype(np.float32)
        
        ### Load the image data
        img = Image.open(imgP)
        if 'L' != img.mode.upper():
            img = img.convert("L")
        img = np.asarray(img, dtype=np.float32)
        return img, gtShape
                
        
class AFLWReader(object):
    @classmethod
    def read(cls, imgPath):
        imgP = imgPath.strip()
        folder, name = os.path.split(imgP)
        file_name,_ = os.path.splitext(name)
        annP = "%s/%s.pts"%(folder, file_name)
        
        ### Load the ground truth of shape
        lines = open(annP, 'r').readlines()
        gtShape = []
        for line in lines:
            line = line.strip()
            if not str.isdigit(line[0]):
                continue
            x, y = line.split()
            gtShape.append((x,y))
            
        gtShape = np.asarray(gtShape, dtype=np.float32)
        
        ### Load the image data
        img = Image.open(imgP)
        if 'L' != img.mode.upper():
            img = img.convert("L")
        img = np.asarray(img, dtype=np.uint8)
        return img, gtShape
        
        
        
   
     
    

    

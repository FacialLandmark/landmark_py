import os
import sys
import numpy as np
from   numpy import loadtxt
from PIL import Image
from affine import *       
import math

### API for augment data
class AugShape(object):
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

class DataWrapper(object):
    def __init__(self):
        self.imgDatas = []
        self.gtShapes = []
        self.bndBoxs  = []        
        self.initShapes = []
        self.meanShape  = None

    def add(self, img, gtShape, bndBox):
        self.imgDatas.append(img)
        self.gtShapes.append(gtShape)
        self.bndBoxs.append(bndBox)
    
    def calMeanShape(self):              
        meanShape = np.zeros(self.gtShapes[0].shape)
        for idx, s in enumerate(self.gtShapes):
            shape = self.shapeReal2Norm(s, self.bndBoxs[idx])
            meanShape = np.add(meanShape, shape)
            
        self.meanShape = meanShape/len(self.gtShapes)

    def genTrainData(self, augNum):
        ### Step1 : Compute the mean shape
        self.calMeanShape()
        
        ### Set meanshape as the initshape
        for bb in self.bndBoxs:
            initShape = self.shapeNorm2Real(self.meanShape,
                                            bb)
            self.initShapes.append(initShape)

        ### Todo : Add augment            
            
        ### Trans list into numpy's array
        self.initShapes = np.asarray(self.initShapes)
        self.gtShapes   = np.asarray(self.gtShapes)
        self.bndBoxs    = np.asarray(self.bndBoxs)         
    
    def shapeReal2Norm(self, realShape, bndBox):
        normShape = np.subtract(realShape, 
                                (bndBox[0],bndBox[1]))
        normShape = np.divide(normShape, 
                              (bndBox[2]-1,bndBox[3]-1))
        return normShape
    
    def shapeNorm2Real(self, normShape, bndBox):
        realShape = np.multiply(normShape, 
                                (bndBox[2]-1, bndBox[3]-1))
        realShape = np.add(realShape, 
                           (bndBox[0],bndBox[1]))
        return realShape
    
    def calResiduals(self):       
        self.residuals = []

        num = self.gtShapes.shape[0]
        for i in range(num):
            ### Project to meanshape coordinary    
            bndBox = self.bndBoxs[i]
            mShape = self.shapeNorm2Real(self.meanShape,
                                         bndBox)             
            T = Affine.fitGeoTrans(shape, mShape)
            err = self.gtShapes[i]-self.initShapes[i]
            err = np.divide(err, (bndBox[2], bndBox[3]))
            err = Affine.transPointsForward(err, T)
            self.residuals.append(err)
        self.residuals = np.asarray(self.residuals)
    
        
class LDReader(object):
    def __init__(self, para): 
        self.dataType = para['dataType']
        self.path = para['path']
        self.augNum = para['augNum']

    def read(self):
        if not os.path.exists(self.path):
            raise Exception("Train set not exist")
        
        dator = DataWrapper()
       
        paths = open(self.path).readlines()
        for imgP in paths:
            try:
                imgP = imgP.strip()
                folder, name = os.path.split(imgP)
                file_name,_ = os.path.splitext(name)
                folder, id_name = os.path.split(folder)
                annP = "%s/Annotations/%s/%s.txt"%(folder,
                                                   id_name,
                                                   file_name)
                ### Load the ground truth of shape
                gtShape=loadtxt(annP, comments="#", 
                                delimiter=",",
                                unpack=False)
                
                ### Load the image data
                img = Image.open(imgP)
                if 'L' != img.mode.upper():
                    img = img.convert("L")
                img = np.asarray(img)

                ### Crop the image
                bndBox = self.getBBoxByPts(gtShape)
                cropB, img = self.cropRegion(bndBox, 2, img)
                gtShape = np.subtract(gtShape, 
                                      (cropB[0], cropB[1]))
                
                ### Get the bndBox. Can use detector Here
                bndBox = self.getBBoxByPts(gtShape)
                dator.add(img, gtShape, bndBox)
            except:
                pass

        ### Calculate the meanShape
        dator.genTrainData(self.augNum)
        return dator

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
        return (x,y,w,h), img[y:y+h, x:x+w]
    
    def printParas(self):
        print('path           = %s'%(self.path))

    
        
   
     
    

    

import numpy as np
import math
import random as RD

### API for shape operation
class Shape(object):
    @classmethod
    def getBBoxByPts(cls, pts):
        maxV = np.max(pts, axis=0)
        minV = np.min(pts, axis=0)
        return (minV[0], minV[1],
                maxV[0]-minV[0]+1,
                maxV[1]-minV[1]+1)    

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
    
    @classmethod
    def augment(cls, shape):
        shape = cls.scale(shape)
        shape = cls.rotate(shape)
        shape = cls.shift(shape)
        return shape

    @classmethod
    def scale(cls, shape):
        scale = 1 + 0.2*(RD.random()-0.5)
        cent = np.mean(shape, axis=0)
        newShape = scale*(shape-cent) + cent
        return newShape

    @classmethod
    def rotate(cls, shape):
        return shape

    @classmethod
    def shift(cls, shape):
        return shape

        
        
        
   
     
    

    

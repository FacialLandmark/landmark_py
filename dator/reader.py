import os
import numpy as np
from   numpy import loadtxt
from PIL import Image

### Different reader for different dataset
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
        
        
        
   
     
    

    

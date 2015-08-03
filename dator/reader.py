import os
import numpy as np
from   numpy import loadtxt
from PIL import Image
import re

class AFWReader(object):
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
            x, y = re.split(',| ', line)
            gtShape.append((x,y))
            
        gtShape = np.asarray(gtShape, dtype=np.float32)
        
        ### Load the image data
        img = Image.open(imgP)
        if 'L' != img.mode.upper():
            img = img.convert("L")
        img = np.asarray(img, dtype=np.uint8)
        return img, gtShape
        
        
        
   
     
    

    

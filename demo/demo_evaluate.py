#!/usr/bin/env python
import os
import sys
import getopt
import time 
import imp
import copy
import numpy as NP
from PIL import Image, ImageDraw

try :
    bIsEnableOpenCV = True
    import cv2
except:
    bIsEnableOpenCV = False

### Add load path
base = os.path.dirname(__file__)
if '' == base:
    base = '.'
sys.path.append('%s/../'%base)

from cascade import *
from utils   import *
from dator   import *

#################################################
### Utils API for evaluate

def usage():
    print("-----------------------------------------------")
    print('[[Usage]]::')
    print('\t./demo.evaluate.py [Paras] train.model path.lst')
    print("[[Paras]]::")
    print("\tshow|s  :  Show the result ")
    print("\thelp|h  :  Print the help information ")
    print("-----------------------------------------------")
    return 

def display(img, gtPnts, resPnts):
    gtPnts = NP.round(gtPnts).astype(NP.int32)
    resPnts = NP.round(resPnts).astype(NP.int32)
    
    showImg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in xrange(gtPnts.shape[0]):
        cv2.circle(showImg, (gtPnts[i, 0], gtPnts[i, 1]),
                   3, (0,0,255), -1)
        cv2.circle(showImg, (resPnts[i, 0], resPnts[i, 1]),
                   3, (255,0,0), -1)
    return showImg

def cropRegion(bbox, scale, img):
    height, width = img.shape
    w = math.floor(scale*bbox[2])
    h = math.floor(scale*bbox[3])
    x = max(0, math.floor(bbox[0]-(w-bbox[2])/2))
    y = max(0, math.floor(bbox[1]-(h-bbox[3])/2))
    w = min(width-x, w)
    h = min(height-y, h)             
    subImg = copy.deepcopy(img[y:y+h, x:x+w])
    return (x,y,w,h), subImg
#################################################

def main(argv):
    try:
        options, args = getopt.getopt(argv, 
                                      "hs", 
                                      ["help", "show"])
    except getopt.GetoptError:  
        usage()
        return
    
    if len(sys.argv) < 2:
        usage()
        return

    bIsShow = False
    for opt , arg in options:
        if opt in ('-h', '--help'):
            usage()
            return
        elif opt in ('-s', '--show'):
            bIsShow = True
    
    try:
        model = args[0]
        imgListPath = args[1]
    except:
        print("ERROR:: Input the 'model' and 'path list'")
        return
        
    cas = LDCascador()  
    cas = cas.loadModel(model)
    
    if True==bIsShow and bIsEnableOpenCV:
        cv2.namedWindow("Landmark")
    
    pathList = open(imgListPath, 'r').readlines()
    reader = AFWReader
    for imgPath in pathList:
        img, gtShape = reader.read(imgPath.strip())

        bndbox = Shape.getBBoxByPts(gtShape)
        cropB, img = cropRegion(bndbox, 3, img)
        gtShape = np.subtract(gtShape, 
                              (cropB[0], cropB[1]))
            
        ### TODO try face detector. 
        bndbox = Shape.getBBoxByPts(gtShape)
        
        ### Set the initial shape
        initShape = Shape.shapeNorm2Real(cas.meanShape,
                                         bndbox)
        
        ### Detect the landmark
        cas.detect(img, bndbox, initShape)        
        if True==bIsShow and bIsEnableOpenCV:
            showImg = display(img, gtShape, initShape)
            cv2.imshow("Landmark", showImg)
            key = cv2.waitKey(1000)
            if key in [ord("q"), 27]:
                break

    if True==bIsShow and bIsEnableOpenCV:
        cv2.destroyAllWindows()
    ### TODO Compute the benchmark
    
if __name__ == '__main__' :
    main(sys.argv[1:])

#!/usr/bin/env python
import os
import sys
import getopt
import time 
import imp
import copy
import numpy as NP
### Add load path
base = os.path.dirname(__file__)
if '' == base:
    base = '.'
sys.path.append('%s/../'%base)

from cascade import *
from utils   import *
from dator   import *

def usage():
    print("-----------------------------------------------")
    print('[[Usage]]::')
    print('\t./demo.evaluate.py [Paras] train.model path.lst')
    print("[[Paras]]::")
    print("\thelp|h     : Print the help information ")
    print("-----------------------------------------------")
    return 

def main(argv):
    try:
        options, args = getopt.getopt(argv, 
                                      "h", 
                                      ["help"])
    except getopt.GetoptError:  
        usage()
        return
    
    if len(sys.argv) < 2:
        usage()
        return

    for opt , arg in options:
        if opt in ('-h', '--help'):
            usage()
            return

    try:
        model = args[0]
        imgListPath = args[1]
    except:
        print("ERROR:: Input the 'model' and 'path list'")
        return
        
    cas = LDCascador()  
    cas = cas.loadModel(model)
    
    pathList = open(imgListPath, 'r').readlines()
    reader = AFLWReader
    for imgPath in pathList:
        img, gtShape = reader.read(imgPath.strip())
        ### Show the ground truth
        gtShape = NP.round(gtShape)
        
        ### Set the initial shape
        initShape = copy.deepcopy(cas.meanShape) 

        ### Set the bndbox. TODO try face detector
        maxV = NP.max(initShape, axis=0)
        minV = NP.min(initShape, axis=0)
        bndbox = (minV[0], minV[1],
                  maxV[0]-minV[0]+1,
                  maxV[1]-minV[1]+1)  
        
        ### Detect the landmark
        cas.detect(img, bndbox, initShape)
        initShape = NP.round(initShape)
        
        ### Show the result
        
    
if __name__ == '__main__' :
    main(sys.argv[1:])

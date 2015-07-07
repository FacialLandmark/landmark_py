import sys
import os
import numpy
import time
from utils   import *
from data import *
from lbfRegressor import *

class LDCascador(object):
    """
    Cascade regression for landmark
    """
    def __init__(self, paras):        
        try:
            self.name     = paras['name']
            self.version  = paras['version']
            self.dataType = paras['dataType']
            self.stageNum = paras['stageNum']
            self.maxTreeNum = paras['maxTreeNum']
            self.treeDepth = paras['treeDepth']
            paras['dataPara']['dataType'] = self.dataType
            self.dataReader = LDReader(paras['dataPara'])
        except:
            raise Exception("Get Paras Failed")
        return

    def printParas(self):
        print('-------------------------------------------')
        print('----------   Configuration    -------------')
        print('name           = %s'%(self.name))
        print('version        = %s'%(self.version))
        print('stageNum       = %s'%(self.stageNum))
        print('maxTreeNum     = %s'%(self.maxTreeNum))
        print('treeDepth      = %s'%(self.treeDepth))
        self.dataReader.printParas()
        print('---------   End of Configuration   --------')
        print('-------------------------------------------\n')
                   
    def train(self, save_path):
        ### mkdir model folder for train model
        if not os.path.exists('%s/model'%(save_path)):
            os.mkdir('%s/model'%(save_path))
        
        ### read data first 
        dator = self.dataReader.read()        

        for stageIdx in xrange(self.stageNum):
            ### train one stage
            pass
        
        self.saveModel(save_path)
        return
    
        
    def loadModel(self, model):
        return
        
    def saveModel(self, save_path):
        name = self.name.lower()
        model_path = "%s/model/landmark.pyobj"%(save_path)
        
    def detect(self, img):
        return 


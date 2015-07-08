import sys
import os
import numpy
import time
from utils   import *
from dator import *
from regressor import *

class LDCascador(object):
    """
    Cascade regression for landmark
    """
    def __init__(self):     
        self.name     = None
        self.version  = None
        self.dataType = numpy.float64
        self.stageNum = None
        
        self.dataWrapper = None
        self.regWrapper = None
        self.regressors = []

    def printParas(self):
        print('-------------------------------------------')
        print('----------   Configuration    -------------')
        print('Name           = %s'%(self.name))
        print('Version        = %s'%(self.version))
        print('Stage Num      = %s'%(self.stageNum))
        print('\n-- Data Config --')
        self.dataWrapper.printParas()
        print('\n-- Regressor Config --')
        self.regWrapper.printParas()
        print('---------   End of Configuration   --------')
        print('-------------------------------------------\n')
                   
    def config(self, paras):
         self.name     = paras['name']
         self.version  = paras['version']
         self.dataType = paras['dataType']
         self.stageNum = paras['stageNum']
         if 'dataType' in paras:
             self.dataType = paras['dataType']

         ### Construct the regressor wrapper
         regPara = paras['regressorPara']
         regPara['dataType'] = self.dataType
         self.regWrapper = RegressorWrapper(regPara)

         ### Construct the data wrapper
         dataPara = paras['dataPara']
         dataPara['dataType'] = self.dataType 
         self.dataWrapper = DataWrapper(dataPara)

    def train(self, save_path):
        ### mkdir model folder for train model
        if not os.path.exists('%s/model'%(save_path)):
            os.mkdir('%s/model'%(save_path))
        
        ### read data first 
        trainSet = self.dataWrapper.read()        

        for idx in xrange(self.stageNum):
            ### train one stage
            reg = self.regWrapper.getClassInstance(idx)
            reg.train()
            self.regressors.append(reg)
        
        self.saveModel(save_path)
    
        
    def loadModel(self, model):
        return
        
    def saveModel(self, save_path):
        name = self.name.lower()
        model_path = "%s/model/landmark.pyobj"%(save_path)
        
    def detect(self, img):
        return 


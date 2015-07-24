import sys
import os
import numpy
import time
import pickle
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
        self.stageNum = None
        
        self.dataWrapper = None
        self.regWrapper = None
        self.regressors = []

    def printParas(self):
        print('------------------------------------------')
        print('----------   Configuration    ------------')
        print('Name           = %s'%(self.name))
        print('Version        = %s'%(self.version))
        print('Stage Num      = %s'%(self.stageNum))
        print('\n-- Data Config --')
        self.dataWrapper.printParas()
        print('\n-- Regressor Config --')
        self.regWrapper.printParas()
        print('---------   End of Configuration   -------')
        print('------------------------------------------\n')
                   
    def config(self, paras):
         self.name     = paras['name']
         self.version  = paras['version']
         self.stageNum = paras['stageNum']

         ### Construct the regressor wrapper
         regPara = paras['regressorPara']
         self.regWrapper = RegressorWrapper(regPara)

         ### Construct the data wrapper
         dataPara = paras['dataPara']
         if 'dataset' in paras:
             dataPara['dataset'] = paras['dataset']
         self.dataWrapper = DataWrapper(dataPara)

    def train(self, save_path):
        ### mkdir model folder for train model
        if not os.path.exists('%s/model'%(save_path)):
            os.mkdir('%s/model'%(save_path))
        
        ### read data first 
        begTime = time.time()
        trainSet = self.dataWrapper.read()        
        t = getTimeByStamp(begTime, 
                           time.time(), 'min')
        print("\tLoading Data: %f mins\n"%(t))

        for idx in xrange(self.stageNum):
            print("\t%drd stage begin ..."%idx)
            begTime = time.time()
            ### calculate the residuals
            trainSet.calResiduals()
            
            ### train one stage
            reg = self.regWrapper.getClassInstance(idx)
            reg.train(trainSet)
            self.regressors.append(reg)        
            
            t = getTimeByStamp(begTime, 
                               time.time(), 'min')
            print("\t%drd stage end : %f mins\n"%(idx, t))
        self.saveModel(save_path)
    
        
    def loadModel(self, model):
        return
        
    def saveModel(self, save_path):
        name = self.name.lower()
        model_path = "%s/model/train.model"%(save_path)
        model = open(model_path, 'w')
        obj_path = "%s/model/%s.pyobj"%(save_path, name)
        model.write(obj_path)

        objFile = open(obj_path, 'wb')
        pickle.dump(self, objFile)
        objFile.close()        
        model.close()
        
    def detect(self, img):
        return 


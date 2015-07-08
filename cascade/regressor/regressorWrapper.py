from  lbfRegressor import *

class RegressorWrapper(object):
    def __init__(self, paras):        
        self.name        = paras['name'].upper()
        self.dataType    = paras['dataType']
        self.maxTreeNums = paras['maxTreeNums']
        self.treeDepths  = paras['treeDepths']

    def printParas(self):        
        print('\tName          = '+self.name)
        print('\tMax Tree Nums = '+str(self.maxTreeNums))
        print('\tTree Depths   = '+str(self.treeDepths))
        
    def getClassInstance(self, idx):
        regPara = dict()        
        regPara['dataType'] = self.dataType

        length = len(self.maxTreeNums)
        __idx = min(idx, length-1)
        regPara['maxTreeNum'] = self.maxTreeNums[__idx]
                     
        length = len(self.treeDepths)
        __idx = min(idx, length-1)
        regPara['treeDepth'] = self.treeDepths[__idx] 
           
        if "LBF_REG"==self.name :
            regClass = LBFRegressor
        else:
            raise Exception("Unsupport: %s "%(self.name))
        
        return regClass(regPara)
            

                     


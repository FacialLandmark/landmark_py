from  lbfRegressor import *

class RegressorWrapper(object):
    def __init__(self, paras):        
        self.name        = paras['name'].upper()
        self.dataType    = paras['dataType']
        self.para        = paras['para']

    def printParas(self):        
        print('\t%-20s= %s'%('name', self.name))
        for key in self.para:
            print('\t%-20s= %s'%(key, str(self.para[key])))
                  
    def getParaLBF(self, idx):
        regPara = dict()        
        regPara['dataType'] = self.dataType

        length = len(self.para['maxTreeNums'])
        _idx = min(idx, length-1)
        regPara['maxTreeNum'] = self.para['maxTreeNums'][_idx]
                     
        length = len(self.para['treeDepths'])
        _idx = min(idx, length-1)
        regPara['treeDepth'] = self.para['treeDepths'][_idx] 

        length = len(self.para['feaNums'])
        _idx = min(idx, length-1)
        regPara['feaNum'] = self.para['feaNums'][_idx] 

        length = len(self.para['radiuses'])
        _idx = min(idx, length-1)
        regPara['radius'] = self.para['radiuses'][_idx]

        length = len(self.para['binNums'])
        _idx = min(idx, length-1)
        regPara['binNum'] = self.para['binNums'][_idx]

        length = len(self.para['feaRanges'])
        _idx = min(idx, length-1)
        regPara['feaRange'] = self.para['feaRanges'][_idx]

        return regPara

    def getClassInstance(self, idx):                   
        if "LBF_REG"==self.name :
            regPara = self.getParaLBF(idx)
            regClass = LBFRegressor
        else:
            raise Exception("Unsupport: %s "%(self.name))
        
        return regClass(regPara)
            

                     


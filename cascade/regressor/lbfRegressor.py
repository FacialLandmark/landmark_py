import sys
import os
import numpy
import time
from utils   import *

class LBFRegressor(object):
    """
    Face Alignment at 3000 FPS via Regressing LBF
    """
    def __init__(self, paras):     
        self.maxTreeNum = paras["maxTreeNum"]
        self.treeDepth  = paras["treeDepth"]

    def train(self):
        pass

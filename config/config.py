import numpy
config = {
    'name'       : "face" ,       
    'version'    : "1.0"  ,
    'dataType'   :  numpy.float32, 
    'stageNum'   :  4 ,
    
    'regressorPara'    :
        {
        'name'       :  'lbf_reg',
        'para'       :
            {
            'maxTreeNums':  [1000, 1000, 500, 500, 400],
            'treeDepths' :  [5],
            'feaNums'    :  [6],
            'radiuses'   :  [0.4, 0.3, 0.2, 0.15, 0.12, 0.1, 0.08],
            'binNums'    :  [511],
            'feaRanges'  :  [[-255, 255]],
            }
        },
    
    'dataPara'   :
        {
        'path': "/home/samuel/data/landmark/path.txt",
        'augNum' : 0
        }     
    }
 

    

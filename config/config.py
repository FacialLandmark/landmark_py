import numpy
config = {
    'name'       : "face" ,      

    ### Different dataset using different reading method
    'dataset'    : "aflw" ,
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
            ### Following para is used to quantize the feature
            'binNums'    :  [511],
            'feaRanges'  :  [[-255, 255]],
            }
        },
    
    'dataPara'   :
        {
        'path': "/home/samuel/data/AFW/path.txt",
        'augNum' : 0
        }     
    }
 

    

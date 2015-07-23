import numpy
config = {
    'name'       : "face_alignment" ,       
    'version'    : "1.0"  ,
    'dataType'   :  numpy.float32, 
    'stageNum'   :  4 ,
    
    'regressorPara'    :
        {
        'name'       :  'lbf_reg',
        'para'       :
            {
            'maxTreeNums':  [50],
            'treeDepths' :  [5],
            'feaNums'    :  [6],
            'radiuses'   :  [0.4, 0.3, 0.2, 0.15],
            'binNums'    :  [511],
            'feaRanges'  :  [[-255, 255]],
            }
        },
    
    'dataPara'   :
        {
        'path': "/home/samuel/project/sandbox/3000fps/image/py.txt",
        'augNum' : 0
        }     
    }
 

    

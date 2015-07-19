import numpy
config = {
    'name'       : "face_alignment" ,       
    'version'    : "1.0"  ,
    'dataType'   :  numpy.float32, 
    'stageNum'   :  5 ,
    
    'regressorPara'    :
        {
        'name'       :  'lbf_reg',
        'para'       :
            {
            'maxTreeNums':  [50],
            'treeDepths' :  [5],
            'feaNums'    :  [300],
            'radiuses'   :  [0.4, 0.3, 0.2, 0.15],
            }
        },
    
    'dataPara'   :
        {
        'path': "/home/samuel/project/sandbox/3000fps/image/py.txt",
        'augNum' : 0
        }     
    }
 

    

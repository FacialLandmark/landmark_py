import numpy
config = {
    'name'       : "face_alignment" ,       
    'version'    : "1.0"  ,
    'dataType'   :  numpy.float32, 
    'stageNum'   :  5 ,
    
    'regressorPara'    :
        {
        'name'       :  'lbf_reg',
        'maxTreeNums':  [1000],
        'treeDepths' :  [5],
        },
    
    'dataPara'   :
        {
        'path': "/home/samuel/project/sandbox/3000fps/image/py.txt",
        'augNum' : 0
        }     
    }
 

    

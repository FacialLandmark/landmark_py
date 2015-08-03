import numpy
config = {
    'name'       : "face" ,      

    ### Different dataset using different reading method
    'dataset'    : "afw" ,
    'version'    : "1.0"  ,
    'stageNum'   :  5 ,
    
    'regressorPara'    :
        {
        'name'       :  'lbf_reg',
        'para'       :
            {
            'maxTreeNums':  [300],
            'treeDepths' :  [5],
            'feaNums'    :  [1000, 750, 500, 375, 250],
            'radiuses'   :  [0.4, 0.3, 0.2, 0.15, 0.12],
            ### Following para is used to quantize the feature
            'binNums'    :  [511],
            'feaRanges'  :  [[-255, 255]],
            }
        },
    
    'dataPara'   :
        {
	### The path.txt file contain the image file list, like following
	###    ./fonder/image1.jpg      
	###    ./fonder/image2.jpg
        ###    ...
        'path': "/home/samuel/project/sandbox/landmark_py/config/afw_train.txt",

        ### augNum < 1 means don't do augmenting 
        'augNum' : 10
        }     
    }
 

    

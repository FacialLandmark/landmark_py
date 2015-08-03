Landmark with Regressition in Python
====

This project aim to implement all the `facial landmark detection with Regressition`. All the following things havd been tested on Ubuntu 14.04.       

* __Supported Algorithms__     
> LBF(Local Binary Features)[1] regression(Based on the [matlab version](https://github.com/jwyang/face-alignment)) 

* __Supported Dataset__    
> [AFW](http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/) dataset. 


#### __Dependencies__    
---    
       
All of the following modules can be easily installed by `pip`    
> [PIL](http://www.pythonware.com/products/pil/)    
> [numpy](http://www.numpy.org/)    
> [scipy](http://www.scipy.org/)    
> [scikit-learn](http://scikit-learn.org/stable/)    
> [OpenCV](http://opencv.org/) (`Optional`: Only Used in demo_evaluate.py to show the result)

Install script on Ubuntu 14.04   
>sudo aptitude install python-pip gfortran     
>sudo pip install pillow numpy scipy sklearn    
>sudo aptitude install python-opencv    


#### __Demo on AFW__    
---    
 
1. Download the AFW dataset [here](http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)
2. Replace the location of afw by yourself in `afw_test.lst` and `afw_train.lst` in config folder(Mine is `/home/samuel/data`)
3. Change `afw_config.py : dataPara.path` by yourself       

* __Train on AFW__     
>./demo_train.py ../config/afw_config.py    

* __Evaluate on AFW__       
>./demo_evaluate.py  ../config/afw_model/train.model  ../config/afw_test.lst       


#### __Tips__    
---    
1. Data Augmentation by flip the image and points      


#### __Extension__
---    
* __Training with your own dataset__    
You should implement your own reader. Please refer to `AFLWReader` in `./cascade/dator/reader.py`.    

* __Implement other regression algorithm__ 
Please refer to `cascade/regressor/lbfRegressor.py`. And then wrapped in `cascade/regressor/regressorWrapper.py`    


#### __TODO__    
---

You can find more todo list via searching "TODO" in source code         
* [__TODO__] Set the shape increment into the tree leaf node. This can speedup the test speed
* [__TODO__] Try random ferns instead of random forest       
* [__TODO__] Try finding the best split when training Random Forest. Now using random split     
* [__TODO__] Try different interpolations when computing `Pixel Difference Feature`      
* [__TODO__] Support `Explicit Shape Regression`    
* [__TODO__] Support `Supervised Descent Method`    


#### __References__    
---    
1. Face Alignment at 3000 FPS via Regressing Local Binary Features    
2. Face Alignment by Explicit Shape Regression    
3. Supervised Descent Method and its Applications to Face Alignment    


#### __Contact__    
---    
If you have any questions, please email `shenfei1208@gmail.com` or creating an issue on GitHub.
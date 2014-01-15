'''
test cases for multiscale pyramids of Convnet features.

    used power_wrapper.py as a starting point.


example usage:
python featpyramid_tests.py --images_file=image_cat.txt --crop_mode=selective_search --model_def=../../../examples/imagenet_deploy.prototxt --pretrained_model=../../../alexnet_train_iter_470000 --output=selective_cat.h5

'''

import numpy as np
import os
import sys
import gflags
import pandas as pd
import time
import skimage.io
import skimage.transform
import selective_search_ijcv_with_python as selective_search
import caffe

#parameters to consider passing to C++ Caffe::featpyramid...
# image filename
# num scales (or something to control this)
# padding amount
# [batchsize is defined in prototxt... fine.]


if __name__ == "__main__":
    imgFname = './pascal_009959.jpg'

    model_def = '../../../examples/imagenet_deploy.prototxt' 
    pretrained_model = '../../../alexnet_train_iter_470000'

    




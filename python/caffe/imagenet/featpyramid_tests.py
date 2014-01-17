'''
test cases for multiscale pyramids of Convnet features.

    used power_wrapper.py as a starting point.

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

#hopefully caffenet is passed by ref...
def test_pyramid_IO(caffenet, imgFname):
    example_np_array = caffenet.testIO() #just return an array with 1 2 3 4...
    #example_np_arrays = caffenet.testIO_multiPlane() #return array of arrays

    print example_np_array
    print example_np_array[0].shape

    caffenet.testString('hi')
    caffenet.testInt(1337)
'''
def featpyramid(caffenet, imgFname, [params]):
    stitched_pyra = caffenet.extract_featpyramid(imgFname, [params])    

    pyra = unstitch_pyra(stitched_pyra) #alternatively, do this in C++ inside extract_featpyramid()

    return pyra
'''

def test_featpyramid(caffenet, imgFname):

    caffenet.extract_featpyramid(imgFname) #test... TODO: return descriptors here 


if __name__ == "__main__":

    #pretend that these flags came off the command line:
    imgFname = './pascal_009959.jpg'
    #model_def = '../../../examples/imagenet_deploy.prototxt'
    model_def = './imagenet_rcnn_batch_1_input_2000x2000_output_conv5.prototxt' 
    pretrained_model = '../../../alexnet_train_iter_470000'
    use_gpu = True
    
    caffenet = caffe.CaffeNet(model_def, pretrained_model)
    caffenet.set_phase_test()
    if use_gpu:
        caffenet.set_mode_gpu()


    #experiments...
    test_pyramid_IO(caffenet, imgFname)

    test_featpyramid(caffenet, imgFname)


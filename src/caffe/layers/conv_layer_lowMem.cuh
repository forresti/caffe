
#ifndef CONV_LAYER_CUH
#define CONV_LAYER_CUH

#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "caffe/blob.hpp"

using namespace std;


namespace caffe {

template <typename Dtype>
void Conv_gpu_lowMem(const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top,
                     int stride, int kernelSize, int num_channels, int height_in, int width_in,
                     int num_output, int height_out, int width_out,
                     int imgID, int numGroups, int groupID);


void hello_cuda();

} // close Caffe class

#endif


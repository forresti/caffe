
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
                     int stride, int channels, int height_in, int width_in,
                     int num_output, int height_out, int width_out,
                     int imgID, int groupID);


} // close Caffe class
#endif


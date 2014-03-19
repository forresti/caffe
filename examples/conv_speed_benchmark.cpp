// Copyright 2013 Yangqing Jia

#include <cuda_runtime.h>
#include <fcntl.h>
#include <google/protobuf/text_format.h>

#include <cstring>
#include <ctime>
#include <cstdio>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/solver.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include <sys/time.h>

using namespace caffe;

double read_timer(){
    struct timeval start;
    gettimeofday( &start, NULL );
    return (double)((start.tv_sec) + 1.0e-6 * (start.tv_usec)) * 1000; //in milliseconds
}

//set up and benchmark layers without actually having a network.
template<typename Dtype>
int conv_speed_test(int num, int channels_in, int height_in, int width_in,
                    int group, int kernelSize, int convStride, int num_output, string niceName)
{
    //TODO: calculate FLOPS based on input size

    //shared_ptr<Blob<Dtype> > blob_bottom(new Blob<Dtype>(num, channels_in, height_in, width_in));
    //shared_ptr<Blob<Dtype> > blob_top(new Blob<Dtype>()); //'top' dims are calculated in ConvolutionLayer::SetUp()

    Blob<Dtype>* blob_bottom_ = new Blob<Dtype>(num, channels_in, height_in, width_in);
    Blob<Dtype>* blob_top_ = new Blob<Dtype>();
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
    blob_bottom_vec_.push_back(blob_bottom_); //ConvolutionLayer likes vectors of blobs.
    blob_top_vec_.push_back(blob_top_);

    LayerParameter layerParams; 
    layerParams.set_kernelsize(kernelSize);
    layerParams.set_stride(convStride);
    layerParams.set_num_output(num_output);
    layerParams.set_group(group);
    layerParams.mutable_weight_filler()->set_type("gaussian");
    layerParams.mutable_bias_filler()->set_type("gaussian");

    ConvolutionLayer<Dtype> convLayer(layerParams);
    convLayer.SetUp(blob_bottom_vec_, &(blob_top_vec_));

    // THE BENCHMARK:
    int num_runs = 10;
    double start = read_timer();
    for (int j = 0; j < num_runs; ++j)
    {
        convLayer.Forward(blob_bottom_vec_, &(blob_top_vec_));
    }
    CUDA_CHECK(cudaDeviceSynchronize()); //for accurate timing
    double layerTime = read_timer() - start; 
    printf("    %s forward: %f ms\n", niceName.c_str(), layerTime/num_runs);
    
    return 0; //TODO: return 1 if error?
}

//TODO: remove unused variables (e.g. num_output?)
template<typename Dtype>
int im2col_speed_test(int num, int channels_in, int height_in, int width_in,
                    int group, int kernelSize, int convStride, int num_output, string niceName)
{
    int height_out = (height_in - kernelSize)/convStride + 1;
    int width_out = (width_in - kernelSize)/convStride + 1;

    Blob<Dtype>* blob_bottom_ = new Blob<Dtype>(num, channels_in, height_in, width_in);
    Blob<Dtype> col_buffer_;
    col_buffer_.Reshape(1, channels_in * kernelSize * kernelSize, height_out, width_out);

    //TODO: check CPU or GPU.
    Dtype* col_data = col_buffer_.mutable_gpu_data();
    const Dtype* bottom_data = blob_bottom_->gpu_data();
    int num_runs = 10;
    double start = read_timer();
    for (int j = 0; j < num_runs; ++j)
    {
        for (int n = 0; n < num; ++n) //each image in the batch
        {
            im2col_gpu(bottom_data + blob_bottom_->offset(n), channels_in, height_in,
                       width_in, kernelSize, convStride, col_data);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize()); //for accurate timing
    double layerTime = read_timer() - start;
    printf("    %s forward: %f ms\n", niceName.c_str(), layerTime/num_runs);
}

int main(int argc, char** argv) {
    cudaSetDevice(0);
    Caffe::set_mode(Caffe::GPU);
    Caffe::set_phase(Caffe::TEST);

//    NetParameter net_param;
//    ReadProtoFromTextFile(argv[1],
//        &net_param);
//    Net<float> caffe_net(net_param);

    int NUM_ = 50;
    
    // alexnet conv1
    conv_speed_test<float>(NUM_, 3, 227, 227, 
                           1, 11, 4, 96, "alexnet conv1");
    im2col_speed_test<float>(NUM_, 3, 227, 227,
                             1, 11, 4, 96, "alexnet im2col1");


    //pool1: stride=2

    conv_speed_test<float>(NUM_, 96, 27, 27,
                           2, 5, 1, 256, "alexnet conv2");
    im2col_speed_test<float>(NUM_, 96, 27, 27,
                           2, 5, 1, 256, "alexnet im2col2");

    //pool2: stride=2

    conv_speed_test<float>(NUM_, 256, 13, 13,
                           1, 3, 1, 384, "alexnet conv3"); //slightly faster than in net_speed_test_forrest (15ms vs 20ms, in GPU mode)
    im2col_speed_test<float>(NUM_, 256, 13, 13,
                           1, 3, 1, 384, "alexnet im2col3"); 

    //there is no pool3

    conv_speed_test<float>(NUM_, 384, 13, 13,
                           2, 3, 1, 384, "alexnet conv4");
    im2col_speed_test<float>(NUM_, 384, 13, 13,
                           2, 3, 1, 384, "alexnet im2col4");

    //there is no pool4

    conv_speed_test<float>(NUM_, 384, 13, 13,
                           2, 3, 1, 256, "alexnet conv5");
    im2col_speed_test<float>(NUM_, 384, 13, 13,
                             2, 3, 1, 256, "alexnet im2col5");

    //TODO: sweep the space of kernelSize, stride, channels, num_output, etc.

    LOG(ERROR) << "*** Benchmark ends ***";

    return 0;
}

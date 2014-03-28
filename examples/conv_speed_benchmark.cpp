// Copyright 2013 Yangqing Jia

#include <cuda_runtime.h>
#include <fcntl.h>
#include <google/protobuf/text_format.h>

#include <cstring>
#include <ctime>
#include <cstdio>
#include <sstream>

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
using namespace std;

double read_timer(){
    struct timeval start;
    gettimeofday( &start, NULL );
    return (double)((start.tv_sec) + 1.0e-6 * (start.tv_usec)) * 1000; //in milliseconds
}

double gflops_to_perform(int num, int channels_in, int height_in, int width_in,
                    int group, int kernelSize, int convStride, int num_output)
{

    double gflops = ((double)height_in * width_in * channels_in * 
                     kernelSize * kernelSize * num_output * num * 2) //*2 is for multiply+add
                     / ((double)convStride * convStride * group * 1e9);

    return gflops;
}

//set up and benchmark layers without actually having a network.
template<typename Dtype>
int conv_speed_test(int num, int channels_in, int height_in, int width_in,
                    int group, int kernelSize, int convStride, int num_output, string niceName)
{
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

//TODO: calculate im2col buf size, and print it out.

    // THE BENCHMARK:
    int num_runs = 10;
    double start = read_timer();
    for (int j = 0; j < num_runs; ++j)
    {
        convLayer.Forward(blob_bottom_vec_, &(blob_top_vec_));
    }
    CUDA_CHECK(cudaDeviceSynchronize()); //for accurate timing
    double layerTime = (read_timer() - start)/num_runs; 
    double gflops_performed = gflops_to_perform(num, channels_in, height_in, width_in,
                                                group, kernelSize, convStride, num_output);
    double gflops_per_sec = gflops_performed / layerTime * 1000; //*1000 for ms to sec 
    LOG(ERROR) << "    " << niceName <<  " forward: " << layerTime << " ms, " << gflops_performed << " gflops ... " << gflops_per_sec << " gflops/sec"; 

    delete blob_bottom_;
    delete blob_top_;
 
    return 0; //TODO: return 1 if error?
}

//TODO: remove unused variables (e.g. num_output and group?)
template<typename Dtype>
int im2col_speed_test(int num, int channels_in, int height_in, int width_in,
                    int group, int kernelSize, int convStride, int num_output, string niceName)
{
    int height_out = (height_in - kernelSize)/convStride + 1;
    int width_out = (width_in - kernelSize)/convStride + 1;

    Blob<Dtype>* blob_bottom_ = new Blob<Dtype>(num, channels_in, height_in, width_in);
    Blob<Dtype> col_buffer_;
    col_buffer_.Reshape(1, channels_in * kernelSize * kernelSize, height_out, width_out);

    Dtype* col_data = NULL;
    const Dtype* bottom_data = NULL;

    int mode = Caffe::mode(); // enum, either 'CPU' or 'GPU'
    if(mode == Caffe::GPU){
        col_data = col_buffer_.mutable_gpu_data();
        bottom_data = blob_bottom_->gpu_data();
    }
    else if(mode == Caffe::CPU){
        col_data = col_buffer_.mutable_cpu_data();
        bottom_data = blob_bottom_->cpu_data();
    } //else unknown mode.
    int num_runs = 10;
    double start = read_timer();
    for (int j = 0; j < num_runs; ++j)
    {
        for (int n = 0; n < num; ++n) //each image in the batch
        {
            if(mode == Caffe::GPU){
                im2col_gpu(bottom_data + blob_bottom_->offset(n), channels_in, height_in,
                           width_in, kernelSize, convStride, col_data);
            }
            else if(mode == Caffe::CPU){
                im2col_cpu(bottom_data + blob_bottom_->offset(n), channels_in, height_in,
                           width_in, kernelSize, convStride, col_data);
            }
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize()); //for accurate timing
    double layerTime = (read_timer() - start) / num_runs;
    double gb_moved = (sizeof(Dtype) * num * channels_in * height_in * width_in / (convStride * convStride)) / 1e9;
    double gb_per_sec = gb_moved / (layerTime / 1000); // 1000 for ms -> sec. 
    LOG(ERROR) << "    " << niceName <<  " forward: " << layerTime << " ms, " << gb_per_sec << " GB/s";
}

//mimic alexnet dims, print out perf results.
void alexnet_speed_test()
{
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
}

// for the configuration below, bigger planes seem to give more gflops/s.
// inputDim=8 and inputDim=16 both take ~20ms.
void vary_input_size(){
    LOG(ERROR) << "running 'vary input size'";

    //experimentally, there doesnt seem to be much pwr-of-2 sensitivity 
    for(int inputDim = 8; inputDim <= 128; inputDim = inputDim*2){ //out of memory if >=128.
        ostringstream niceName;
        niceName << "inputDim = " << inputDim << ".";

        conv_speed_test<float>(50, 384, inputDim, inputDim,                           
                               2, 3, 1, 256, niceName.str());
    }
}

//3x3 filter is as good as bigger filters in terms of gflops/s (~1700 gflops/s with 55x55 planes.)
void vary_filter_size(){
    LOG(ERROR) << "running 'vary filter size'";
    for(int filterSize=1; filterSize<10; filterSize++) //out of memory if >10
    { 
        ostringstream niceName;
        niceName << "filterSize = " << filterSize << ".";

        conv_speed_test<float>(50, 384, 55, 55, 
                               2, filterSize, 1, 256, niceName.str());
    }
}

void vary_channels_in(){
    LOG(ERROR) << "running 'num input channels'";
    for(int channels_in=4; channels_in <= 2048; channels_in=channels_in*2) //
    { 
        ostringstream niceName;
        niceName << "channels_in = " << channels_in << ".";

        conv_speed_test<float>(50, channels_in, 55, 55, 
                               2, 3, 1, 256, niceName.str());
    }
}

void vary_batch_size()
{
    LOG(ERROR) << "running 'num batch size'";
    for(int NUM_=1; NUM_<60; NUM_+=4)
    { 
        ostringstream niceName;
        niceName << "NUM_ = " << NUM_ << ".";

        conv_speed_test<float>(NUM_, 384, 55, 55, 
                               2, 3, 1, 256, niceName.str());
    }
}

void vary_num_groups()
{
    LOG(ERROR) << "running 'num groups'";
    for(int group=1; group<=8; group=group*2)
    { 
        ostringstream niceName;
        niceName << "num groups = " << group << ".";

        conv_speed_test<float>(50, 384, 55, 55, 
                               group, 3, 1, 256, niceName.str());
    }
}

void vary_num_filters()
{
    LOG(ERROR) << "running 'num filters'";
    for(int num_output = 2; num_output < 10000; num_output=num_output*2)
    { 
        ostringstream niceName;
        niceName << "num filters = " << num_output << ".";

        conv_speed_test<float>(50, 384, 55, 55, 
                               2, 3, 1, num_output, niceName.str());
    }
}

int main(int argc, char** argv) {
    ::google::InitGoogleLogging(argv[0]);
    cudaSetDevice(0);
    Caffe::set_mode(Caffe::GPU);
    Caffe::set_phase(Caffe::TEST);

    //alexnet_speed_test();
    vary_num_filters();
    vary_num_groups();
    vary_batch_size();
    vary_channels_in();
    vary_input_size();
    vary_filter_size();

    return 0;
}

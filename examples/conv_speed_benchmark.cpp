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
#include <sys/time.h>

using namespace caffe;

double read_timer(){
    struct timeval start;
    gettimeofday( &start, NULL );
    return (double)((start.tv_sec) + 1.0e-6 * (start.tv_usec)) * 1000; //in milliseconds
}

template <typename Dtype>
class ConvolutionLayer_notProtected : protected ConvolutionLayer<Dtype> 
{
    //using ConvolutionLayer<Dtype>::ConvolutionLayer; //inherit constructors

    public:
      // hack around 'protected' Forward_cpu in ConvolutionLayer
      void Forward_cpu_notProtected(const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top){
          Forward_cpu(bottom, top);
      }
};

//set up and benchmark layers without actually having a network.
template<typename Dtype>
int conv_speed_test(int num, int channels_in, int height_in, int width_in,
                    int group, int kernelSize, int convStride, int num_output)
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
    //ConvolutionLayer_notProtected<Dtype> convLayer(layerParams);

    convLayer.SetUp(blob_bottom_vec_, &(blob_top_vec_));

    //TODO add timer and do multiple runs.

    convLayer.Forward(blob_bottom_vec_, &(blob_top_vec_));

    return 0; //TODO: return 1 if error?
}

int main(int argc, char** argv) {
  cudaSetDevice(0);
  Caffe::set_mode(Caffe::CPU);
  Caffe::set_phase(Caffe::TEST);
  int repeat = 5;

  NetParameter net_param;
  ReadProtoFromTextFile(argv[1],
      &net_param);
  Net<float> caffe_net(net_param);


  // alexnet conv1
  conv_speed_test<float>(1, 3, 227, 227, 
                         1, 11, 4, 96);

#if 0
  // Run the network without training.
  LOG(ERROR) << "Performing Forward";
  // Note that for the speed benchmark, we will assume that the network does
  // not take any input blobs.
  caffe_net.Forward(vector<Blob<float>*>());

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();


  LOG(ERROR) << "*** Benchmark begins ***";
  printf("  avg time per layer: \n");
  for (int i = 0; i < layers.size(); ++i) {
    const string& layername = layers[i]->layer_param().name();
    //clock_t start = clock();
    double start = read_timer();
    for (int j = 0; j < repeat; ++j) {
      layers[i]->Forward(bottom_vecs[i], &top_vecs[i]);
    }
    CUDA_CHECK(cudaDeviceSynchronize()); //for accurate timing
    double layerTime = read_timer() - start; 
    //printf("    %s forward: %f ms\n", layername.c_str(), layerTime); 
    printf("    %s forward: %f ms\n", layername.c_str(), layerTime/repeat); 


  }
#endif

  LOG(ERROR) << "*** Benchmark ends ***";

  return 0;
}

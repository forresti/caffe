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
#include <sys/time.h>

using namespace caffe;

double read_timer(){
    struct timeval start;
    gettimeofday( &start, NULL );
    return (double)((start.tv_sec) + 1.0e-6 * (start.tv_usec)) * 1000; //in milliseconds
}

//easy-to-parse version of layer dimensions
class LayerSummary{
  public:
    int layerHeight;
    int layerWidth;
    int filterHeight;
    int filterWidth;
    int nFilters; //num_outpu for this layer
    int depth; //== prev layer's nFilters
    string name;
};

int main(int argc, char** argv) {
  cudaSetDevice(0);
  Caffe::set_mode(Caffe::GPU);
  Caffe::set_phase(Caffe::TRAIN);
  int repeat = 5;

  NetParameter net_param;
  ReadProtoFromTextFile(argv[1],
      &net_param);
  Net<float> caffe_net(net_param);

  // Run the network without training.
  LOG(ERROR) << "Performing Forward";
  // Note that for the speed benchmark, we will assume that the network does
  // not take any input blobs.
  caffe_net.Forward(vector<Blob<float>*>());
  //LOG(ERROR) << "Performing Backward";
  //LOG(ERROR) << "Initial loss: " << caffe_net.Backward();

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();

  //parse layer dims (only conv layers, for now)
  //vector<LayerSummary*> layerSummaries = getLayerSummaries(layers);

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
#if 0
  for (int i = layers.size() - 1; i >= 0; --i) {
    const string& layername = layers[i]->layer_param().name();
    clock_t start = clock();
    for (int j = 0; j < repeat; ++j) {
      layers[i]->Backward(top_vecs[i], true, &bottom_vecs[i]);
    }
    LOG(ERROR) << layername << "\tbackward: "
        << float(clock() - start) / CLOCKS_PER_SEC << " seconds.";
  }
#endif
  LOG(ERROR) << "*** Benchmark ends ***";

  //TODO: free layerSummaries
  return 0;
}

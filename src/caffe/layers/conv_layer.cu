// Copyright 2013 Yangqing Jia

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

#include "conv_layer_lowMem.cuh" //yuck ... should have a header for this, but the make system doesn't notice *.cuh files

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "Conv Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "Conv Layer takes a single blob as output.";
  KSIZE_ = this->layer_param_.kernelsize();
  STRIDE_ = this->layer_param_.stride();
  GROUP_ = this->layer_param_.group();
  NUM_ = bottom[0]->num();
  CHANNELS_ = bottom[0]->channels();
  HEIGHT_ = bottom[0]->height();
  WIDTH_ = bottom[0]->width();
  NUM_OUTPUT_ = this->layer_param_.num_output();
  CHECK_GT(NUM_OUTPUT_, 0);
  CHECK_EQ(CHANNELS_ % GROUP_, 0);
  // The im2col result buffer would only hold one image at a time to avoid
  // overly large memory usage.
  HEIGHT_OUT_ = (HEIGHT_ - KSIZE_) / STRIDE_ + 1;
  WIDTH_OUT_ = (WIDTH_ - KSIZE_) / STRIDE_ + 1;
  col_buffer_.Reshape(1, CHANNELS_ * KSIZE_ * KSIZE_, HEIGHT_OUT_, WIDTH_OUT_);
  // Set the parameters
  CHECK_EQ(NUM_OUTPUT_ % GROUP_, 0)
      << "Number of output should be multiples of group.";
  biasterm_ = this->layer_param_.biasterm();
  // Figure out the dimensions for individual gemms.
  M_ = NUM_OUTPUT_ / GROUP_;
  K_ = CHANNELS_ * KSIZE_ * KSIZE_ / GROUP_;
  N_ = HEIGHT_OUT_ * WIDTH_OUT_;
  (*top)[0]->Reshape(bottom[0]->num(), NUM_OUTPUT_, HEIGHT_OUT_, WIDTH_OUT_);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (biasterm_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Intialize the weight
    this->blobs_[0].reset(
        new Blob<Dtype>(NUM_OUTPUT_, CHANNELS_ / GROUP_, KSIZE_, KSIZE_));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(
        GetFiller<Dtype>(this->layer_param_.weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (biasterm_) {
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, NUM_OUTPUT_));
      shared_ptr<Filler<Dtype> > bias_filler(
          GetFiller<Dtype>(this->layer_param_.bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // Set up the bias filler
  if (biasterm_) {
    bias_multiplier_.reset(new SyncedMemory(N_ * sizeof(Dtype)));
    Dtype* bias_multiplier_data =
        reinterpret_cast<Dtype*>(bias_multiplier_->mutable_cpu_data());
    for (int i = 0; i < N_; ++i) {
        bias_multiplier_data[i] = 1.;
    }
  }
};

//pseudo-code:
#if 0 
template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu_lowMem(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  Dtype* col_data = col_buffer_.mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;

  /*
  //ORIGINAL CODE:
  for (int n = 0; n < NUM_; ++n) {
    // First, im2col
    im2col_cpu(bottom_data + bottom[0]->offset(n), CHANNELS_, HEIGHT_,
        WIDTH_, KSIZE_, STRIDE_, col_data);
    // Second, innerproduct with groups
    for (int g = 0; g < GROUP_; ++g) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
        (Dtype)1., weight + weight_offset * g, col_data + col_offset * g,
        (Dtype)0., top_data + (*top)[0]->offset(n) + top_offset * g);
    }
    */

    // First, transpose input data with 'channels' as the fast dimension
    //     bottom_data(img, group, channel, y, x) -> bottom_data_trans(group, img, y, x, channel) //row-major notation

    //bottom_data_trans = transpose(bottom_data); //TODO

    // Second, transpose filters ("weights") with 'channels' as the fast dimension
    //     weights(filterID, channels, y, x) -> weights_trans(y, x, filterID, channels) //row-major notation

    //weight_trans = transpose(weight); //TODO

    // Third, multiply by each (x,y) location in filters. 
    //     (one BLAS call per (x,y) location in filters)

    for (int g = 0; g < GROUP_; ++g){
        for(int filterY=0; filterY<kernelSize; filterY++){
            for(int filterX=0; filterX<kernelSize; filterX++){
                
                //A = filters, "weight_trans"
                //B = input data, "bottom_data_trans"

                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,  
                                     NUM_OUTPUT_, //M: rows in A
                                     NUM_ * HEIGHT_OUT_ * WIDTH_OUT_, //N: cols in B
                                     CHANNELS_ / GROUP_, //K: cols in A == rows in B 
                                     (Dtype)1., //alpha
                                     weights_trans, //A
                                     TODO, //lda = rows in A (unless I'm using padding) 
                                     bottom_data_trans, //B
                                     TODO, //ldb = rows in B (not trans(B)) -- remember, col major.
                                     (Dtype)1., //beta (using 1 means we +=C instead of overwriting C)
                                     top_data_trans, //C
                                     TODO /* ldc = rows in C */ );
 
            }
        }
    } 

    // Fourth, un-transpose output data (might be avoidable with clever BLAS settings)
    //     top_data(group, img, y, x, channel) ->  top_data(img, group, channel, y, x) //row-major notation

    // Fifth, add bias
    if (biasterm_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, NUM_OUTPUT_,
          N_, 1, (Dtype)1., this->blobs_[1]->cpu_data(),
          reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()),
          (Dtype)1., top_data + (*top)[0]->offset(n));
    }
  }

}
#endif

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  Dtype* col_data = col_buffer_.mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;
  for (int n = 0; n < NUM_; ++n) {
    // First, im2col
    im2col_cpu(bottom_data + bottom[0]->offset(n), CHANNELS_, HEIGHT_,
        WIDTH_, KSIZE_, STRIDE_, col_data);
    // Second, innerproduct with groups
    for (int g = 0; g < GROUP_; ++g) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
        (Dtype)1., weight + weight_offset * g, col_data + col_offset * g,
        (Dtype)0., top_data + (*top)[0]->offset(n) + top_offset * g);
    }
    // third, add bias
    if (biasterm_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, NUM_OUTPUT_,
          N_, 1, (Dtype)1., this->blobs_[1]->cpu_data(),
          reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()),
          (Dtype)1., top_data + (*top)[0]->offset(n));
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {

  //bool use_low_mem_conv = true; //TODO: expose this to user
  bool use_low_mem_conv = false;

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;

  Dtype* col_data = NULL;
  if(!use_low_mem_conv)
  {
    //init extra buffer for BLAS
    col_data = col_buffer_.mutable_gpu_data(); //only allocated when touched
  }

  for (int n = 0; n < NUM_; ++n) 
  {
    if(use_low_mem_conv)
    {
      for(int g = 0; g < GROUP_; ++g) 
      {
        Conv_gpu_lowMem<Dtype>(bottom, top, weight, 
                        STRIDE_, KSIZE_, CHANNELS_, HEIGHT_, WIDTH_,
                        NUM_OUTPUT_, HEIGHT_OUT_, WIDTH_OUT_,
                        n, GROUP_, g);
      }
    }

    else{ //use BLAS
      // First, im2col
      im2col_gpu(bottom_data + bottom[0]->offset(n), CHANNELS_, HEIGHT_,
          WIDTH_, KSIZE_, STRIDE_, col_data);

      // Second, innerproduct with groups
      for (int g = 0; g < GROUP_; ++g) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
          (Dtype)1., weight + weight_offset * g, col_data + col_offset * g,
          (Dtype)0., top_data + (*top)[0]->offset(n) + top_offset * g);
        //CUDA_CHECK(cudaDeviceSynchronize()); //for speed tests
      }
    }

    // third, add bias
    if (biasterm_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, NUM_OUTPUT_,
          N_, 1, (Dtype)1., this->blobs_[1]->gpu_data(),
          reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
          (Dtype)1., top_data + (*top)[0]->offset(n));
    }
  }
}

template <typename Dtype>
Dtype ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  Dtype* col_data = col_buffer_.mutable_cpu_data();
  Dtype* col_diff = col_buffer_.mutable_cpu_diff();
  // bias gradient if necessary
  Dtype* bias_diff = NULL;

  if (biasterm_) {
    bias_diff = this->blobs_[1]->mutable_cpu_diff();
    memset(bias_diff, 0, sizeof(Dtype) * this->blobs_[1]->count());
    for (int n = 0; n < NUM_; ++n) {
      caffe_cpu_gemv<Dtype>(CblasNoTrans, NUM_OUTPUT_, N_,
          1., top_diff + top[0]->offset(n),
          reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()), 1.,
          bias_diff);
    }
  }

  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;
  memset(weight_diff, 0, sizeof(Dtype) * this->blobs_[0]->count());
  for (int n = 0; n < NUM_; ++n) {
    // since we saved memory in the forward pass by not storing all col data,
    // we will need to recompute them.
    im2col_cpu(bottom_data + (*bottom)[0]->offset(n), CHANNELS_, HEIGHT_,
        WIDTH_, KSIZE_, STRIDE_, col_data);
    // gradient w.r.t. weight. Note that we will accumulate diffs.
    for (int g = 0; g < GROUP_; ++g) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
        (Dtype)1., top_diff + top[0]->offset(n) + top_offset * g,
        col_data + col_offset * g, (Dtype)1.,
        weight_diff + weight_offset * g);
    }
    // gradient w.r.t. bottom data, if necessary
    if (propagate_down) {
      for (int g = 0; g < GROUP_; ++g) {
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
          (Dtype)1., weight + weight_offset * g,
          top_diff + top[0]->offset(n) + top_offset * g,
          (Dtype)0., col_diff + col_offset * g);
      }
      // col2im back to the data
      col2im_cpu(col_diff, CHANNELS_, HEIGHT_,
        WIDTH_, KSIZE_, STRIDE_, bottom_diff + (*bottom)[0]->offset(n));
    }
  }
  return Dtype(0.);
}

template <typename Dtype>
Dtype ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->gpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  Dtype* col_data = col_buffer_.mutable_gpu_data();
  Dtype* col_diff = col_buffer_.mutable_gpu_diff();
  // bias gradient if necessary
  Dtype* bias_diff = NULL;

  if (biasterm_) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
    CUDA_CHECK(cudaMemset(bias_diff, 0,
        sizeof(Dtype) * this->blobs_[1]->count()));
    for (int n = 0; n < NUM_; ++n) {
      caffe_gpu_gemv<Dtype>(CblasNoTrans, NUM_OUTPUT_, N_,
          1., top_diff + top[0]->offset(n),
          reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
          1., bias_diff);
    }
  }

  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;
  CUDA_CHECK(cudaMemset(weight_diff, 0,
      sizeof(Dtype) * this->blobs_[0]->count()));
  for (int n = 0; n < NUM_; ++n) {
    // since we saved memory in the forward pass by not storing all col data,
    // we will need to recompute them.
    im2col_gpu(bottom_data + (*bottom)[0]->offset(n), CHANNELS_, HEIGHT_,
        WIDTH_, KSIZE_, STRIDE_, col_data);
    // gradient w.r.t. weight. Note that we will accumulate diffs.
    for (int g = 0; g < GROUP_; ++g) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
        (Dtype)1., top_diff + top[0]->offset(n) + top_offset * g,
        col_data + col_offset * g, (Dtype)1.,
        weight_diff + weight_offset * g);
    }
    // gradient w.r.t. bottom data, if necessary
    if (propagate_down) {
      for (int g = 0; g < GROUP_; ++g) {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
          (Dtype)1., weight + weight_offset * g,
          top_diff + top[0]->offset(n) + top_offset * g,
          (Dtype)0., col_diff + col_offset * g);
      }
      // col2im back to the data
      col2im_gpu(col_diff, CHANNELS_, HEIGHT_,
          WIDTH_, KSIZE_, STRIDE_, bottom_diff + (*bottom)[0]->offset(n));
    }
  }
  return Dtype(0.);
}

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe

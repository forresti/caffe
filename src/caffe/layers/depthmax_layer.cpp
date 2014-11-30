#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

// we comupute the top-k elements  over slices of the depth (across channels) for all (x,y) positions in a layer.
// then, we zero out all but the top-k elements in the channels of each (x,y) position

//argmax_layer is a bit different -- it takes the top-k argmax across all elements in a layer (e.g. "get the indices of top-5 categories")


//testing:
//  ./build/tools/caffe time -model=toy_alexnet_depthMax.prototxt
//  TODO: dedicated test file

/*
bool depthmax_sort(my_pair& left, my_pair& right)
{
  return left.first < right.first;
}
*/

template <typename Dtype>
void DepthMaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int top_k_ = this->layer_param_.depthmax_param().top_k(); //number of channel indicies to preserve (zero out the rest.)

  printf("in DepthMaxLayer::Forward_cpu(). top_k = %d \n", top_k_);

  //TODO: assert that top_data and bottom_data are the same array?

  //TODO: think about the 'group' stuff... any reason to think this would interact poorly with fake multi-GPU stuff?

  for(int b=0; b<bottom[0]->num(); b++) { //batch
    for(int y=0; y<bottom[0]->height(); y++) {
      for(int x=0; x<bottom[0]->width(); x++) {
        //channel_vector = bottom[b, :, y, x]. all channels in current location.
        std::vector<std::pair<Dtype, int> > channel_vector( bottom[0]->channels() ); //preallocate to len(channels)
        int base_idx = (b * bottom[0]->channels() * bottom[0]->height() * bottom[0]->width()) +
                       (y * bottom[0]->width()) + x;
        for (int c = 0; c < bottom[0]->channels(); c++) {
          channel_vector[c] = 
              std::make_pair(bottom_data[base_idx + (c * bottom[0]->height() * bottom[0]->width())], c);
        }
        //sort channels from highest to lowest value, and remember their original positions in the array 
        std::partial_sort(channel_vector.begin(), channel_vector.begin() + top_k_,
                          channel_vector.end(), std::greater<std::pair<Dtype, int> >());

#if 0
        printf("  (b=%d, y=%d, x=%d), top k channels:", b, y, x);
        for(int k=0; k<top_k_; k++){
          printf("    [channel=%d, value=%f] ", channel_vector[k].second, channel_vector[k].first);
        }
        printf("\n");
#endif

        //zero out all but top_k elements per channel
        for(int c = 0; c < bottom[0]->channels(); c++) {

          //is this channel in the top k?
          bool preserve_channel = false;
          for(int k=0; k<top_k_; k++){
            if(channel_vector[k].second == c)
              preserve_channel = true;
          }

          //if channel isn't in top-k, then zero it out.
          if(!preserve_channel){
            top_data[base_idx + (c * bottom[0]->height() * bottom[0]->width())] = 0;
          }
        }
      }
    }
  }




  /*
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }
  */
}

template <typename Dtype>
void DepthMaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    /*
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
    */
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(DepthMaxLayer);


}  // namespace caffe

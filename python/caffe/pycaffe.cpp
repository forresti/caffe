// Copyright Yangqing Jia 2013
// pycaffe provides a wrapper of the caffe::Net class as well as some
// caffe::Caffe functions so that one could easily call it from Python.
// Note that for python, we will simply use float as the data type.

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include "caffe/caffe.hpp"

// Temporary solution for numpy < 1.7 versions: old macro.
#ifndef NPY_ARRAY_C_CONTIGUOUS
#define NPY_ARRAY_C_CONTIGUOUS NPY_C_CONTIGUOUS
#endif


using namespace caffe;
using boost::python::extract;
using boost::python::len;
using boost::python::list;
using boost::python::object;


// A simple wrapper over CaffeNet that runs the forward process.
struct CaffeNet
{
  CaffeNet(string param_file, string pretrained_param_file) {
    net_.reset(new Net<float>(param_file));
    net_->CopyTrainedLayersFrom(pretrained_param_file);
  }

  virtual ~CaffeNet() {}

  inline void check_array_against_blob(
      PyArrayObject* arr, Blob<float>* blob) {
    CHECK(PyArray_FLAGS(arr) & NPY_ARRAY_C_CONTIGUOUS);
    CHECK_EQ(PyArray_NDIM(arr), 4);
    CHECK_EQ(PyArray_ITEMSIZE(arr), 4);
    npy_intp* dims = PyArray_DIMS(arr);
    CHECK_EQ(dims[0], blob->num());
    CHECK_EQ(dims[1], blob->channels());
    CHECK_EQ(dims[2], blob->height());
    CHECK_EQ(dims[3], blob->width());
  }

  // The actual forward function. It takes in a python list of numpy arrays as
  // input and a python list of numpy arrays as output. The input and output
  // should all have correct shapes, are single-precisionabcdnt- and c contiguous.
  void Forward(list bottom, list top) {
    vector<Blob<float>*>& input_blobs = net_->input_blobs();
    CHECK_EQ(len(bottom), input_blobs.size());
    CHECK_EQ(len(top), net_->num_outputs());
    // First, copy the input
    for (int i = 0; i < input_blobs.size(); ++i) {
      object elem = bottom[i];
      PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(elem.ptr());
      check_array_against_blob(arr, input_blobs[i]);
      switch (Caffe::mode()) {
      case Caffe::CPU:
        memcpy(input_blobs[i]->mutable_cpu_data(), PyArray_DATA(arr),
            sizeof(float) * input_blobs[i]->count());
        break;
      case Caffe::GPU:
        cudaMemcpy(input_blobs[i]->mutable_gpu_data(), PyArray_DATA(arr),
            sizeof(float) * input_blobs[i]->count(), cudaMemcpyHostToDevice);
        break;
      default:
        LOG(FATAL) << "Unknown Caffe mode.";
      }  // switch (Caffe::mode())
    }
    //LOG(INFO) << "Start";
    const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled();
    //LOG(INFO) << "End";
    for (int i = 0; i < output_blobs.size(); ++i) {
      object elem = top[i];
      PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(elem.ptr());
      check_array_against_blob(arr, output_blobs[i]);
      switch (Caffe::mode()) {
      case Caffe::CPU:
        memcpy(PyArray_DATA(arr), output_blobs[i]->cpu_data(),
            sizeof(float) * output_blobs[i]->count());
        break;
      case Caffe::GPU:
        cudaMemcpy(PyArray_DATA(arr), output_blobs[i]->gpu_data(),
            sizeof(float) * output_blobs[i]->count(), cudaMemcpyDeviceToHost);
        break;
      default:
        LOG(FATAL) << "Unknown Caffe mode.";
      }  // switch (Caffe::mode())
    }
  }

  void Backward(list top_diff, list bottom_diff) {
    vector<Blob<float>*>& output_blobs = net_->output_blobs();
    vector<Blob<float>*>& input_blobs = net_->input_blobs();
    CHECK_EQ(len(bottom_diff), input_blobs.size());
    CHECK_EQ(len(top_diff), output_blobs.size());
    // First, copy the output diff
    for (int i = 0; i < output_blobs.size(); ++i) {
      object elem = top_diff[i];
      PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(elem.ptr());
      check_array_against_blob(arr, output_blobs[i]);
      switch (Caffe::mode()) {
      case Caffe::CPU:
        memcpy(output_blobs[i]->mutable_cpu_diff(), PyArray_DATA(arr),
            sizeof(float) * output_blobs[i]->count());
        break;
      case Caffe::GPU:
        cudaMemcpy(output_blobs[i]->mutable_gpu_diff(), PyArray_DATA(arr),
            sizeof(float) * output_blobs[i]->count(), cudaMemcpyHostToDevice);
        break;
      default:
        LOG(FATAL) << "Unknown Caffe mode.";
      }  // switch (Caffe::mode())
    }
    //LOG(INFO) << "Start";
    net_->Backward();
    //LOG(INFO) << "End";
    for (int i = 0; i < input_blobs.size(); ++i) {
      object elem = bottom_diff[i];
      PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(elem.ptr());
      check_array_against_blob(arr, input_blobs[i]);
      switch (Caffe::mode()) {
      case Caffe::CPU:
        memcpy(PyArray_DATA(arr), input_blobs[i]->cpu_diff(),
            sizeof(float) * input_blobs[i]->count());
        break;
      case Caffe::GPU:
        cudaMemcpy(PyArray_DATA(arr), input_blobs[i]->gpu_diff(),
            sizeof(float) * input_blobs[i]->count(), cudaMemcpyDeviceToHost);
        break;
      default:
        LOG(FATAL) << "Unknown Caffe mode.";
      }  // switch (Caffe::mode())
    }
  }

  //void testIO(){ } //dummy example

  //float* -> numpy -> boost python (which can be returned to Python)
  boost::python::object array_to_boostPython_4d(float* pyramid_float, 
                                                int nbPlanes, int depth_, int MaxHeight_, int MaxWidth_)
  {

    npy_intp dims[4] = {nbPlanes, depth_, MaxHeight_, MaxWidth_}; //in floats
    PyArrayObject* pyramid_float_py = (PyArrayObject*)PyArray_New( &PyArray_Type, 4, dims, NPY_FLOAT, 0, pyramid_float, 0, 0, 0 ); //not specifying strides

    //thanks: stackoverflow.com/questions/19185574 
    boost::python::object pyramid_float_py_boost(boost::python::handle<>((PyObject*)pyramid_float_py));
    return pyramid_float_py_boost;
  } 

  //return a list containing one 4D numpy/boost array. (toy example)
  boost::python::list testIO()
  {
    int nbPlanes = 1;
    int depth_ = 1;
    int MaxHeight_ = 10;
    int MaxWidth_ = 10;

    //prepare data that we'll send to Python
    float* pyramid_float = (float*)malloc(sizeof(float) * nbPlanes * depth_ * MaxHeight_ * MaxWidth_);
    memset(pyramid_float, 0, sizeof(float) * nbPlanes * depth_ * MaxHeight_ * MaxWidth_);
    pyramid_float[10] = 123; //test -- see if it shows up in Python

    boost::python::object pyramid_float_py_boost = array_to_boostPython_4d(pyramid_float, nbPlanes, depth_, MaxHeight_, MaxWidth_);

    boost::python::list blobs_top_boost; //list to return
    blobs_top_boost.append(pyramid_float_py_boost); //put the output array in list

    return blobs_top_boost; //compile error: return-statement with no value  
  }

  void testString(string st){
    printf("    string from python: %s \n", st.c_str());
  }

  void testInt(int i){
    printf("    int from python: %d \n", i);
  }



  // The caffe::Caffe utility functions.
  void set_mode_cpu() { Caffe::set_mode(Caffe::CPU); }
  void set_mode_gpu() { Caffe::set_mode(Caffe::GPU); }
  void set_phase_train() { Caffe::set_phase(Caffe::TRAIN); }
  void set_phase_test() { Caffe::set_phase(Caffe::TEST); }
  void set_device(int device_id) { Caffe::SetDevice(device_id); }

  // The pointer to the internal caffe::Net instant.
	shared_ptr<Net<float> > net_;
};

// The boost python module definition.
BOOST_PYTHON_MODULE(pycaffe)
{
  import_array(); //setup numpy. (this is finnicky: stackoverflow.com/questions/12253389)
  //boost::python::converter::registry::insert( &CaffeNet::testIO, type_id<PyArrayObject>()); //try to help testIO to return an object?

  boost::python::class_<CaffeNet>(
      "CaffeNet", boost::python::init<string, string>())
      .def("Forward",         &CaffeNet::Forward)
      .def("Backward",        &CaffeNet::Backward)
      .def("set_mode_cpu",    &CaffeNet::set_mode_cpu)
      .def("set_mode_gpu",    &CaffeNet::set_mode_gpu)
      .def("set_phase_train", &CaffeNet::set_phase_train)
      .def("set_phase_test",  &CaffeNet::set_phase_test)
      .def("set_device",      &CaffeNet::set_device)
      .def("testIO",          &CaffeNet::testIO) //Forrest's test (return a numpy array)
      .def("testString",      &CaffeNet::testString) 
      .def("testInt",         &CaffeNet::testInt)
  ;
}

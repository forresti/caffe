#!/usr/bin/env sh

#syntax: test_net.bin test_net_config trained_net_model n_iter GPU/CPU [deviceID]
#        (deviceID is optional)

GLOG_logtostderr=1 ../build/examples/test_net.bin imagenet_val.prototxt ../examples/alexnet_train_iter_470000 1000 GPU


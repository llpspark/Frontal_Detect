#!/usr/bin/env sh
CAFFEROOT=/home/spark/grocery/caffe_tmp
./$1 /home/spark/grocery/FER/codes/caffe_2classify/test/size32_2classify_deploy.prototxt /home/spark/grocery/FER/codes/caffe_2classify/model/size32_2classify_v5_iter_2040.caffemodel labels.txt > res.txt 
#LD_LIBRARY_PATH=$CAFFEROOT/build/lib/:$LD_LIBRARY_PATH ./$1 /home/spark/grocery/FER/codes/caffe_2classify/test/size32_2classify_deploy.prototxt /home/spark/grocery/FER/codes/caffe_2classify/model/size32_2classify_v4_iter_19500.caffemodel labels.txt 

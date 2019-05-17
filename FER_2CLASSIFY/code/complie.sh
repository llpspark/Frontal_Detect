#!/usr/bin/env sh
CAFFEROOT=/home/spark/grocery/caffe_tmp
export PKG_CONFIG_PATH=/usr/local/opencv-3/lib/pkgconfig/
echo "compile "$1
g++ -o $1.bin $1.cpp  -std=c++11 -O3 -D CPU_ONLY -I $CAFFEROOT/include/ -I $CAFFEROOT/build/src/ -L $CAFFEROOT/build/lib -lcaffe -lglog -lboost_system -lprotobuf `pkg-config --cflags --libs opencv` -L/usr/lib/x86_64-linux-gnu/ -ltiff 
#g++ -o $1.bin $1.cpp  -std=c++11 -O3 -D CPU_ONLY -I $CAFFEROOT/include/ -I $CAFFEROOT/.build_release/src/ -L $CAFFEROOT/build/lib -lcaffe -lglog -lboost_system -lprotobuf `pkg-config --cflags --libs opencv` -L/usr/lib/x86_64-linux-gnu/ -ltiff 

echo "done"

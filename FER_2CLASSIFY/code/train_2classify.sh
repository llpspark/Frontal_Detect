#PRETRAINED_MODEL=/home/spark/grocery/FER/codes/caffe_2classify/model/size32_2classify_v3_iter_49500.caffemodel
t=$(date +%Y-%m-%d_%H:%M:%S) 
LOG=../log/size32_2classify_v4_$t.log
GLOG_logtostderr=1 /home/spark/caffe-master/build/tools/caffe train \
    --solver=./size32_2classify_solver.prototxt \
    --gpu=3 2>&1 | tee $LOG
    #--weights=$PRETRAINED_MODEL \

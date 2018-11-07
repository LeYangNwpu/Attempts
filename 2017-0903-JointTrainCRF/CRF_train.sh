GLOG_logtostderr=0 GLOG_log_dir=./Log/ \
/disk1/yangle/CVPR2018/software/caffe-master/.build_release/tools/caffe train \
  --solver=/disk1/yangle/CVPR2018/code/TrainNet/JointTrainCRF/CRF_solver.prototxt \
  -gpu 3 \
  --weights /disk1/yangle/CVPR2018/code/TrainNet/JointTrainCRF/pretrain_encoder_decoder.caffemodel \
  2>&1 | tee /disk1/yangle/CVPR2018/code/TrainNet/JointTrainCRF/CRF_train_doc.txt



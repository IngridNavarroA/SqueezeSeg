#!/bin/bash

export GPUID=0
export NET="squeezeSeg"
export IMAGE_SET="train"
export LOG_DIR="./log/"
export DATA_DIR="./data/"
export CLASSES="ng"
export RES=0
export TEST=1
export STEPS=60000
export CRF=0

if [ $# -eq 0 ]
then
      echo "Usage: ./scripts/train.sh [options]"
      echo " "
      echo "options:"
      echo "-h, --help              show brief help"
      echo "-gpu                    gpu id | n (cpu)"
      echo "-image_set              [train|val]"
      echo "-steps                  Number of training steps."
      echo "-test                   Test to execute [1 - 8]. Test [1-4] - VLP16, Test [5-8] VLP32"
      echo "-restore                Restore from checkpoint or train from scratch"
  exit 0
fi

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "Usage: ./scripts/train.sh [options]"
      echo " "
      echo "options:"
      echo "-h, --help              show brief help"
      echo "-gpu                    gpu id | n (cpu)"
      echo "-image_set              [train|val]"
      echo "-steps                  Number of training steps."
      echo "-test                   Test to execute [1 - 8]. Test [1-4] - VLP16, Test [5-8] VLP32"
      echo "-restore                Restore from checkpoint or train from scratch"
      exit 0
      ;;
    -gpu)
      export GPUID="$2"
      shift
      shift
      ;;
    -image_set)
      export IMAGE_SET="$2"
      shift
      shift
      ;;
    -steps)
      export STEPS="$2"
      shift
      shift
      ;;
    -res)
      export RES="$2"
      shift
      shift
      ;;
    -test)
     export TEST="$2"
     shift
     shift
     ;;
    *)
      break
      ;;
  esac
done

# Test 1 - VLP 16, CRF, classes: unknown, car, pedestrian and cyclist
# Test 2 - VLP 16, CRF, classes: unknown, car, pedestrian, cyclist and ground
# Test 3 - VLP 16, no CRF, classes: unknown, car, pedestrian and cyclist 
# Test 4 - VLP 16, no CRF, classes: unknown, car, pedestrian, cyclist and ground

if [ $TEST -eq 1 ]
then
   NET="squeezeSeg16"
   LOG_DIR="./log/log16_NG_CRF"
   DATA_DIR="./data/ng_vlp16/"
   CLASSES="ng"
   CRF=1
elif [ $TEST -eq 2 ]
then 
   NET="squeezeSeg16"
   LOG_DIR="./log/log16_G_CRF"
   DATA_DIR="./data/g_vlp16/"
   CLASSES="ext"
   CRF=1
elif [ $TEST -eq 3 ]
then
   NET="squeezeSeg16x"
   LOG_DIR="./log/log16_NG_NOCRF"
   DATA_DIR="./data/ng_vlp16/"
   CLASSES="ng"
   CRF=0
elif [ $TEST -eq 4 ]
then
   NET="squeezeSeg16x"
   LOG_DIR="./log/log16_G_NOCRF"
   DATA_DIR="./data/g_vlp16/"
   CLASSES="ext"
   CRF=0
fi

logdir="$LOG_DIR"

python ./src/train.py \
   --dataset=KITTI \
   --pretrained_model_path=./data/SqueezeNet/squeezenet_v1.1.pkl \
   --data_path=$DATA_DIR \
   --image_set=$IMAGE_SET \
   --net=$NET \
   --max_steps=$STEPS \
   --gpu=$GPUID \
   --classes=$CLASSES \
   --restore=$RES \
   --train_dir="$logdir/train" \
   --summary_step=50 \
   --checkpoint_step=500
   
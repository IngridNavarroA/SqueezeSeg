#!/bin/bash

export GPUID=0
export NET="squeezeSeg"
export IMAGE_SET="train"
export LOG_DIR="./log/"
export STEPS=25000
export DATA_DIR="./data/"
export EXT="n"
export RES="n"

if [ $# -eq 0 ]
then
  echo "Usage: ./scripts/train.sh [options]"
  echo " "
  echo "options:"
  echo "-h, --help                show brief help"
  echo "-gpu                      gpu id"
  echo "-net                      NArchitecture for vlp64, vlp32 or vlp16"
  echo "-image_set                (train|val)"
  echo "-log_dir                  Where to save logs."
  echo "-steps                    Number of training steps."
  echo "-data_dir                 Where the data to train is."
  echo "-ext                      Train with additional class."
  echo "-res                      Train from checkpoint."
  exit 0
fi

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "Usage: ./scripts/train.sh [options]"
      echo " "
      echo "options:"
      echo "-h, --help                show brief help"
      echo "-gpu                      gpu id"
      echo "-net                      NArchitecture for vlp64, vlp32 or vlp16"
      echo "-image_set                (train|val)"
      echo "-log_dir                  Where to save logs."
      echo "-steps                    Number of training steps."
      echo "-data_dir                 Where the data to train is."
      echo "-ext                      Train with additional class."
      echo "-res                      Train from checkpoint."
      exit 0
      ;;
    -gpu)
      export GPUID="$2"
      shift
      shift
      ;;
    -net)
      export NET="$2"
      shift
      shift
      ;;
    -image_set)
      export IMAGE_SET="$2"
      shift
      shift
      ;;
    -log_dir)
      export LOG_DIR="$2"
      shift
      shift
      ;;
    -steps)
      export STEPS="$2"
      shift
      shift
      ;;
    -data_dir)
      export DATA_DIR="$2"
      shift
      shift
      ;;
    -ext)
      export EXT="$2"
      shift
      shift
      ;;
    -res)
      export RES="$2"
      shift
      shift
      ;;
    *)
      break
      ;;
  esac
done

logdir="$LOG_DIR"

python ./src/train.py \
  --dataset=KITTI \
  --pretrained_model_path=./data/SqueezeNet/squeezenet_v1.1.pkl \
  --data_path=$DATA_DIR \
  --image_set=$IMAGE_SET \
  --train_dir="$logdir/train" \
  --net=$NET \
  --max_steps=$STEPS \
  --summary_step=50 \
  --checkpoint_step=500 \
  --gpu=$GPUID \
  --extended=$EXT \
  --restore=$RES

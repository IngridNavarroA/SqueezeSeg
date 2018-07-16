#!/bin/bash

export GPUID=0
export NET="squeezeSeg"
export IMAGE_SET="val"
export LOG_DIR="./log/"
export DATA_DIR="./data/"
export EXT="y"
export RES="n"

if [ $# -eq 0 ]
then
  echo "Usage: ./scripts/eval.sh [options]"
  echo " "
  echo "options:"
  echo "-h, --help                show brief help"
  echo "-gpu                      gpu id | n (cpu)"
  echo "-image_set                (train|val)"
  echo "-net                      Network architecture to train"
  echo "-log_dir                  Where to load models and save logs."
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
      echo "-net                      Network architecture to train"
      echo "-image_set                (train|val)"
      echo "-log_dir                  Where to load models and save logs."
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
traindir="$logdir/train/"
valdir="$logdir/eval_$IMAGE_SET"

python ./src/eval.py \
  --dataset=KITTI \
  --data_path=$DATA_DIR/ \
  --image_set=$IMAGE_SET \
  --eval_dir="$valdir" \
  --checkpoint_path="$traindir" \
  --net=$NET \
  --gpu=$GPUID \
  --extended=$EXT \
  --restore=$RES

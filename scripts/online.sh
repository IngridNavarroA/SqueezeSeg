#!/bin/bash

export SUB_TOPIC="/kitti/points_raw"

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "Usage: ./scripts/online.sh [options]"
      echo " "
      echo "options:"
      echo "-h, --help                show brief help"
      echo "-sub_topic                subscribe point cloud topic, default '/kitti/points_raw'."
      exit 0
      ;;
    -sub_topic)
      export SUB_TOPIC="$2"
      shift
      shift
      ;;
    *)
      break
      ;;
  esac
done

bash ./scripts/quickstart.sh

python ./src/online.py --sub_topic "$SUB_TOPIC"

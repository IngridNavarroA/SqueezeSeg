#!/bin/bash

get_char() {
  SAVEDSTTY=`stty -g`
  stty -echo
  stty cbreak
  dd if=/dev/tty bs=1 count=1 2> /dev/null
  stty -raw
  stty echo
  stty $SAVEDSTTY
}

export RVIZ_CFG="./rviz/default.rviz"

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "Usage: ./scripts/quickstart.sh [options]"
      echo " "
      echo "options:"
      echo "-h, --help                show brief help"
      echo "-rviz_cfg                 rviz configure file, default './rviz/default.rviz'."
      exit 0
      ;;
    -rviz_cfg)
      export RVIZ_CFG="$2"
      shift
      shift
      ;;
    *)
      echo "Usage: ./scripts/quickstart.sh [options]"
      echo " "
      echo "options:"
      echo "-h, --help                show brief help"
      echo "-rviz_cfg                 rviz configure file, default './rviz/default.rviz'."
      exit 0
      ;;
  esac
done

echo ""
echo "HOSTNAME"
echo "======="
echo `hostname`

# start roscore if not running
if ! pidof -x "roscore" >/dev/null; then
    echo ""
    echo "ROSCORE STARTING"
    echo "======="
    setsid roscore 1>/dev/null 2>&1 &
fi

echo "Waiting for roscore ready..."
sleep 1
echo "Press any key to continue!"
char=`get_char`

# start rviz if not running
if ! pidof -x "rviz" >/dev/null; then
    echo ""
    echo "RVIZ STARTING"
    echo "======="
    setsid rosrun rviz rviz -d $RVIZ_CFG 1>/dev/null 2>&1 &
fi
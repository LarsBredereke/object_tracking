#!/bin/bash

# pip environment goes here
source ~/bins/hiwi/bin/activate

LOGDIR=logs/`date +"%F_%T"`
mkdir $LOGDIR

# kill subtasks if Pipeline is terminated prematurely
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

export TF_CPP_MIN_LOG_LEVEL=3 # stop tensorflow spam

# object tracking pipeline
echo "starting annotate_corners.py for object tracking"
python3 annotate_corners.py &> "$LOGDIR/annotate_corners_log.txt"
echo "starting yolo.py for object tracking"
python3 yolo.py &> "$LOGDIR/yolo_log.txt"
echo "starting camera_params.py for object tracking"
python3 camera_params.py &> "$LOGDIR/camera_params_log.txt"
echo "starting ekf.py for object tracking"
python3 ekf.py &> "$LOGDIR/ekf_log.txt"

wait
echo "all done"
#!/bin/bash

# activate pip managed venv
source ~/venvs/tsd1/bin/activate

# alternatively for yolo.py on Rechenknechte, use micromamba for venv
# # >>> mamba initialize >>>
# # !! Contents within this block are managed by 'mamba init' !!
# export MAMBA_EXE='/share/temp/students/lbredereke/mamba/bin/micromamba';
# export MAMBA_ROOT_PREFIX='/home/labr/micromamba';
# __mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
# if [ $? -eq 0 ]; then
#     eval "$__mamba_setup"
# else
#     alias micromamba="$MAMBA_EXE"  # Fallback on help from mamba activate
# fi
# unset __mamba_setup
# # <<< mamba initialize <<<

# micromamba activate ot

LOGDIR=~/repos/tsd-one/03_Process/tsd1_pipeline2024/logs/`date +"%F_%T"`
mkdir $LOGDIR
mkdir "$LOGDIR/ekf"

# kill subtasks if Pipeline is terminated prematurely
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

export TF_CPP_MIN_LOG_LEVEL=3 # stop tensorflow spam

echo "starting path_discovery"
python3 path_discovery.py &> "$LOGDIR/path_discovery_log.txt"
if [ $? -eq 0 ]; then
    echo "path_discovery ok"
else
    echo "path_discovery fail"
    exit
fi

echo "starting calc_bounds"
python3 calc_bounds.py --n_jobs 4 &> "$LOGDIR/calc_bounds_log.txt"
if [ $? -eq 0 ]; then
    echo "calc_bounds ok"
else
    echo "calc_bounds fail"
    exit
fi

echo "starting gaze"
python3 gaze.py &> "$LOGDIR/gaze_log.txt"&

echo "starting eeg"
python3 eeg.py &> "$LOGDIR/eeg_log.txt"&

echo "starting plux"
python3 plux.py &> "$LOGDIR/plux_log.txt"&

echo "starting audio"
python3 audio.py &> "$LOGDIR/audio_log.txt"&

echo "starting annotations"
python3 annotations.py &> "$LOGDIR/annotations_log.txt"&

echo "starting mocap"
python3 mocap.py &> "$LOGDIR/mocap_log.txt"&

echo "starting mocap matched"
python3 mocap_recovered.py &> "$LOGDIR/mocap_recovered_log.txt"&

echo "starting webcams"
python3 webcams.py --n_jobs 4 &> "$LOGDIR/webcams_log.txt"&

echo "starting head_cams"
python3 head_cams.py &> "$LOGDIR/head_cams_log.txt"&

wait # the following scripts rely on files generated in above code

echo "starting whisper"
python3 whisper_annotation.py &> "$LOGDIR/whisper_annotaion_log.txt"&

echo "starting notes"
python3 note.py &> "$LOGDIR/note_log.txt"&

# # object tracking pipeline is executed sequentially
# echo "starting yolo.py for object tracking"
# python3 yolo.py &> "$LOGDIR/yolo_log.txt"
# echo "starting camera_params.py for object tracking"
# python3 camera_params.py &> "$LOGDIR/camera_params_log.txt"
# for i in $(seq 1 100); # memory leak: restart script for every session
# do
#     echo "starting ekf.py for object tracking for session $i"
#     python3 ekf.py $i &> "$LOGDIR/ekf/session_$i.txt"
# done

wait
echo "all done"
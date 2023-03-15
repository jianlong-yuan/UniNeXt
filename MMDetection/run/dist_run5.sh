export PYTHONPATH=$PWD:$PYTHONPAT
bash tools/dist_train_dlc.sh \
/mnt/workspace/linfangjian.lfj/mmdetection/configs/ours/ours_lastglobal_ws13.py 8

# pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
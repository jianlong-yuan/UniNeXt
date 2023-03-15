export PYTHONPATH=$PWD:$PYTHONPAT
bash tools/dist_train_dlc.sh \
/mnt/workspace/linfangjian.lfj/mmdetection/configs/ours/ours_lastglobal_ws14.py 8

# export PYTHONPATH=$PWD:$PYTHONPAT
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# bash tools/dist_train.sh \
# /mnt/workspace/linfangjian.lfj/mmdetection/configs/ours/ours_cswin_3x.py 8
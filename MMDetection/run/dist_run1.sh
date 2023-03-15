# export PYTHONPATH=$PWD:$PYTHONPAT
# bash tools/dist_train_dlc.sh \
# /mnt/workspace/linfangjian.lfj/mmdetection/configs/ours/ours_lastglobal_ws11_3x.py 8
cd /mnt/workspace/linfangjian.lfj/mmsegmentation
python tools/get_flops.py /mnt/workspace/linfangjian.lfj/mmsegmentation/configs/setr/setr_vit-large_naive_8x1_768x768_80k_cityscapes.py --shape 640 640

# export PYTHONPATH=$PWD:$PYTHONPAT
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# bash tools/dist_train.sh \
# /mnt/workspace/linfangjian.lfj/mmdetection/configs/ours/ours_cswin_3x.py 8
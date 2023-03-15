# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# bash tools/dist_test.sh \
# /mnt/workspace/linfangjian.lfj/mmsegmentation/configs/ours/upernet_ours.py \
# /mnt/workspace/linfangjian.lfj/mmsegmentation/work_dirs/upernet_ours/iter_160000.pth \
# 8 \
# --aug-test --eval mIoU


CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash tools/dist_test.sh \
/mnt/workspace/linfangjian.lfj/mmsegmentation/configs/ours/UniNeXt_S.py \
/mnt/workspace/linfangjian.lfj/mmsegmentation/work_dirs/UniNeXt_S/iter_132000.pth \
4 \
--aug-test --eval mIoU


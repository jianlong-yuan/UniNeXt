CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash train.sh 8 --data /earth-nas/datasets/imagenet-1k/ \
--model cswin_small \
-b 64 --lr 5e-4 --weight-decay .05 --amp --img-size 224 --warmup-epochs 20 --model-ema-decay 0.99984 --drop-path 0.4

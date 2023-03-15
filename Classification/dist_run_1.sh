PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# Total 32 GPUs
# For UniNext-CSWinAttention-Base / UniNext-LocalAttention-Base
python -m torch.distributed.launch \
    --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $(dirname "$0")/dist_main_1.py \
    --data /earth-nas/datasets/imagenet-1k/ \
    --model cswin_base \
#   --model UniNeXt_B \
    -b 32 --lr 1e-3 --weight-decay .1 --amp --img-size 224 --warmup-epochs 20 --model-ema-decay 0.99992 --drop-path 0.5

# For UniNext-CSWinAttention-Small / UniNext-LocalAttention-Small
# python -m torch.distributed.launch \
#     --nnodes=$WORLD_SIZE \
#     --node_rank=$RANK \
#     --nproc_per_node=1 \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
#     $(dirname "$0")/dist_main_1.py \
#     --data /earth-nas/datasets/imagenet-1k/ \
#     --model cswin_small \
# #   --model UniNeXt_S \
#     -b 32 --lr 1e-3 --weight-decay .05 --amp --img-size 224 --warmup-epochs 20 --model-ema-decay 0.99984 --drop-path 0.4

# For UniNext-CSWinAttention-Tiny / UniNext-LocalAttention-Tiny
# python -m torch.distributed.launch \
#     --nnodes=$WORLD_SIZE \
#     --node_rank=$RANK \
#     --nproc_per_node=1 \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
#     $(dirname "$0")/dist_main_1.py \
#     --data /earth-nas/datasets/imagenet-1k/ \
#     --model cswin \
# #   --model UniNeXt_T \
#     -b 32 --lr 1e-3 --weight-decay .05 --amp --img-size 224 --warmup-epochs 20 --model-ema-decay 0.99984 --drop-path 0.2

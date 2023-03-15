# #!/bin/bash
# if false
# then
#         source /opt/rh/devtoolset-7/enable
#         pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
# else
#         echo "instll None"
# fi



if true
then
        bash /mnt/workspace/linfangjian.lfj/mmdetection/run/dist_run3.sh
else
        echo "wait"
fi
cd /mnt/workspace/DCNv2
bash make.sh

if true
then
        cd /mnt/workspace/DLC/DCNv2
        bash make.sh
        cd /mnt/workspace/linfangjian.lfj/DilatedFormer
else
        echo "install None"
fi
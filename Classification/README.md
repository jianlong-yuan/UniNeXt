# UniNeXt: Exploring A Unified Architecture for Vision Recognition


## Requirements

timm==0.3.4, pytorch>=1.4, opencv, ... , run:

```
bash install_req.sh
```

Apex for mixed precision training is used for finetuning. To install apex, run:

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Data prepare: ImageNet with the following folder structure, you can extract imagenet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

## Train

Train the three lite variants with DLC: UniNeXt-Tiny, UniNeXt-Small and UniNeXt-Base:
```
bash run/connetct.sh
```
Train the three lite variants with DSW: UniNeXt-Tiny, UniNeXt-Small and UniNeXt-Base:
```
bash run.sh
```


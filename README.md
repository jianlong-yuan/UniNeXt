UniNeXt: Exploring A Unified Architecture for Vision Recognition

## Main Results on ImageNet

| model | Attention |  acc@1 | #params | #FLOPs | 
|:---:  | :---:    | :---: |   :---: | :---: |
| UniNeXt-T | local window | 83.6 | 24M  | 4.3G  |  
| UniNeXt-S | local window| 84.1 | 51M  | 9.5G  |
| UniNeXt-B | local window | 84.4 | 91M  | 17.1G  |    
| UniNeXt-T | cross-shaped window | 83.5 | 24M  | 4.3G  |  
| UniNeXt-S | cross-shaped window | 84.3 | 51M  | 9.6G  |   
| UniNeXt-B | cross-shaped window | 84.7 | 91M  | 17.2G  |  

## Main Results on Downstream Tasks


**COCO Object Detection**

| backbone | Attention | Method | pretrain | lr Schd | box mAP | mask mAP |
|:---:     | :---:  | :---:  |  :---:   | :---:   |   :---: | :---:    |
| UniNeXt-T | local window | Mask R-CNN | ImageNet-1K | 1x | 48.6 | 43.4 |
| UniNeXt-S | local window | Mask R-CNN | ImageNet-1K | 1x | 49.0 | 43.7 |
| UniNeXt-B | local window | Mask R-CNN | ImageNet-1K | 1x | 49.3 | 43.9 |
| UniNeXt-T | cross-shaped window | Mask R-CNN | ImageNet-1K | 1x | 48.7 | 43.6 |
| UniNeXt-S | cross-shaped window | Mask R-CNN | ImageNet-1K | 1x | 49.2 | 43.8 |
| UniNeXt-B | cross-shaped window | Mask R-CNN | ImageNet-1K | 1x | - | - |

**ADE20K Semantic Segmentation (val)**

| Backbone | Attention | Method | pretrain | Crop Size | Lr Schd | mIoU | mIoU (ms+flip) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| UniNeXt-T | local window | UPerNet | ImageNet-1K | 512x512 | 160K | 49.7 | 50.6 |
| UniNeXt-S | local window | UperNet | ImageNet-1K | 512x512 | 160K | 51.0 | 51.8 |
| UniNeXt-B | local window | UperNet | ImageNet-1K | 512x512 | 160K | 51.4 | 52.2 | 
| UniNeXt-T | cross-shaped window | UPerNet | ImageNet-1K | 512x512 | 160K | 49.9 | - |
| UniNeXt-S | cross-shaped window | UperNet | ImageNet-1K | 512x512 | 160K | 51.5 | - | 
| UniNeXt-B | cross-shaped window | UperNet | ImageNet-1K | 512x512 | 160K | 51.6 | - | 

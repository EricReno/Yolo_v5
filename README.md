# Yolo_v5 (You only look once)

- Yolo_v5 pytorch源码，含训练、推理、评测、部署脚本
- 细节请参考Yolo_v4论文描述：
https://arxiv.org/abs/2004.10934 (v4)

## Train Tricks

|Epoch|BatchSize|Pretrain|WarmUp|Datasets|Train|Val  |Augment |Model_Ema|
|:---:|:---:    |:---:   |:---: |:---:   |:---:|:---:|:---:   |:---:    |
|300  |64       |CoCo    |3     |VOC20   |16553|4952 |hsv+Ablu|True     |

|NMS_TH.|EvalIou_TH.|Confidence_TH.|Optimizer|Grad_accu|Lr_sche|LearningRate|
|:---:  |:---:      |:---:         |:---:    |:---:    |:---:  |:---:       |
|0.7    |0.5        |0.001         |adamw    |1        |linear |0.001       |

## Results:

|Backbone    |Size |Params|FLOPs |Model  |Params|FLOPs  |Mosaic|Mixup|mAP(%)|
|:---:       |:---:|:---: |:---: |:---:  |:---: |:---:  |:---: |:---:|:---: |
|cspdarknet_n|512  | 1.06M| 1.90G|Yolov5n| 2.26M|  4.48G|1.0   |False|73.37 | 27M|
|cspdarknet_t|512  | 2.37M| 4.07G|Yolov5t| 5.05M| 15.34G|1.0   |False|      |
|cspdarknet_s|512  | 4.21M| 7.06G|Yolov5s| 8.96M| 17.22G|1.0   |False|79.76 |103M|
|cspdarknet_m|512  |12.35M|21.55G|Yolov5m|25.32M| 47.32G|1.0   |0.15 |      |
|cspdarknet_l|512  |27.08M|48.72G|Yolov5l|54.20M| 99.82G|1.0   |0.15 |      |
|cspdarknet_x|512  |50.30M|92.60G|Yolov5x|99.06M|180.73G|1.0   |0.15 |--    |

<table>

<tr>
<th>Yolov5n</th>
<!-- <th>Yolov5t</th> -->
<th>Yolov5s</th>
<!-- <th>Yolov5m</th> -->
</tr>

<tr>
<td>

|ClassName  |AP   |
|--         |--   |
|aeroplane  |0.798|
|bicycle    |0.794|
|bird       |0.690|
|boat       |0.658|
|bottle     |0.605|
|bus        |0.809|
|car        |0.848|
|cat        |0.794|
|chair      |0.581|
|cow        |0.794|
|diningtable|0.684|
|dog        |0.741|
|horse      |0.826|
|motorbike  |0.788|
|person     |0.797|
|pottedplant|0.489|
|sheep      |0.754|
|sofa       |0.680|
|train      |0.809|
|tvmonitor  |0.736|
|mAP        |0.734|

</td>
<td>
    
|ClassNames |AP   |
|--         |--   |
|aeroplane  |0.841|
|bicycle    |0.861|
|bird       |0.780|
|boat       |0.711|
|bottle     |0.676|
|bus        |0.848|
|car        |0.881|
|cat        |0.848|
|chair      |0.658|
|cow        |0.866|
|diningtable|0.763|
|dog        |0.826|
|horse      |0.886|
|motorbike  |0.847|
|person     |0.838|
|pottedplant|0.590|
|sheep      |0.824|
|sofa       |0.757|
|train      |0.853|
|tvmonitor  |0.800|
|mAP        |0.798|

</td>

</tr> 
</table>

<video src="" 
       controls 
       width="100%" 
       height="auto" 
       style="max-width: 720px; height: auto; display: block; object-fit: contain;">
</video>
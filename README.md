# Yolo_v5 (You only look once)

- 以Yolo_v5 基础的火焰、烟雾检测算法

训练数据：990张
测试数据：110张
yolov5n模型的精度如下：

## Results:

|Backbone    |Size |Params|FLOPs |Model  |Params|FLOPs  |Mosaic|Mixup|mAP(%)|
|:---:       |:---:|:---: |:---: |:---:  |:---: |:---:  |:---: |:---:|:---: |
|cspdarknet_n|512  | 1.06M| 1.90G|Yolov5n| 2.26M|  4.48G|1.0   |False|76.57 | 27M|
|cspdarknet_t|512  | 2.37M| 4.07G|Yolov5t| 5.05M| 15.34G|1.0   |False|      |
|cspdarknet_s|512  | 4.21M| 7.06G|Yolov5s| 8.96M| 17.22G|1.0   |False|      |


<table>

<tr>
<th>Yolov5n</th>
</tr>

<tr>
<td>

|ClassName  |AP   |
|--         |--   |
|fire       |0.774|
|smoke      |0.757|
|           |     |
|mAP        |0.766|

</td>

</tr> 
</table>

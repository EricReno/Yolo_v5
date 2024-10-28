# Yolo_v5 (You only look once)

- 以Yolo_v5 基础的电动车进电梯检测算法

训练数据：87张
测试数据：11张
yolov5n模型的精度如下：

## Results:

|Backbone    |Size |Params|FLOPs |Model  |Params|FLOPs  |Mosaic|Mixup|mAP(%)|
|:---:       |:---:|:---: |:---: |:---:  |:---: |:---:  |:---: |:---:|:---: |
|cspdarknet_n|512  | 1.06M| 1.90G|Yolov5n| 2.26M|  4.48G|1.0   |False|89.35 | 27M|
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
|person     |0.711|
|bicycle    |1.000|
|motorcycle |0.970|
|           |     |
|mAP        |0.893|

</td>

</tr> 
</table>

import os
import sys
import onnx
import torch
sys.path.append('../')
from config import parse_args
from model.yolov1 import YOLOv1

args = parse_args()

x = torch.randn(1, 3, 640, 640)

model = YOLOv1(args = args, 
               device = torch.device('cpu'),
               trainable = False,
               nms_thresh = args.nms_thresh,
               conf_thresh = args.conf_thresh)
model.deploy = True

ckpt_path = os.path.join(args.root, args.project, 'results', '82.pth')
ckpt_state_dict = torch.load(ckpt_path, map_location='cpu')["model"]
model.load_state_dict(ckpt_state_dict)
model = model.eval()


with torch.no_grad():
    torch.onnx.export(
        model,
        x,
        "yolo_v1.onnx",
        opset_version=11,
        input_names=['input'],
        output_names=['output'])

onnx_model = onnx.load("yolo_v1.onnx") 
onnx.save(onnx.shape_inference.infer_shapes(onnx_model), "yolo_v1.onnx")

try: 
    onnx.checker.check_model(onnx_model) 
except Exception: 
    print("Model incorrect") 
else: 
    print("Model correct")
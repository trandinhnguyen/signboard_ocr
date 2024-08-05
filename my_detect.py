import cv2
import torch
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadImages, letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device
import sys
import os
__dir__ = os.path.dirname(os.path.abspath(__file__))
print(__dir__)
exit()
def detect(
    source,
    weights=["weights/best.pt"],
    img_size=640,
    conf_thres=0.25,
    iou_thres=0.45,
):
    # Initialize
    # set_logging()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = select_device(device)
    half = device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    img_size = check_img_size(img_size, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    # Run inference
    if device.type != "cpu":
        model(
            torch.zeros(
                1,
                3,
                img_size,
                img_size,
            )
            .to(device)
            .type_as(next(model.parameters()))
        )  # run once

    img0 = cv2.imread(source)
    # Padded resize
    img = letterbox(img0, img_size, stride=stride)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(
        pred,
        conf_thres,
        iou_thres,
        classes=None,
        agnostic=False,
    )

    # Process detections
    for det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(
                img.shape[2:],
                det[:, :4],
                img0.shape,
            ).round()

            print(det)
            for *xyxy, conf, cls in reversed(det):
                print(xyxy)
        else:
            return None

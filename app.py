import cv2
import torch
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import pickle

import sys

sys.path.insert(0, "yolov7/")
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords

from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

# Load model yolov7
yolov7 = attempt_load(["yolov7/weights/best.pt"], map_location="cpu")  # load FP32 model
stride = int(yolov7.stride.max())  # model stride


print("loading paddle ocr")
paddle_ocr = PaddleOCR(
    use_angle_cls=True,
    # lang="vi",
    rec_model_dir="paddleocr_data/inference/v3_latin_mobile_generated_dataset_v2",
    rec_char_dict_path="paddleocr_data/vietnamese_dict.txt",
    # use_gpu=False,
)  # need to run only once to download and load model into memory
print("paddle ocr loaded")

config = Cfg.load_config_from_file("vietocr_data/config/base.yml")
config.update(Cfg.load_config_from_file("vietocr_data/config/vgg-seq2seq.yml"))
config["weights"] = "vietocr_data/vgg_seq2seq.pth"
config["device"] = "cpu"
detector = Predictor(config)


from_disk = pickle.load(open("gru_model/tv_layer.pkl", "rb"))
text_vec_layer = tf.keras.layers.TextVectorization(split="character")
text_vec_layer.set_weights(from_disk["weights"])
labels = ["Shop's name", "Address", "Phone number"]
gru_model = tf.keras.models.load_model("gru_model")


def detect(
    img0,
    img_size=640,
    conf_thres=0.25,
    iou_thres=0.45,
):
    img_size = check_img_size(img_size, s=stride)  # check img_size
    # Padded resize
    img = letterbox(img0, img_size, stride=stride)[0]
    # Convert
    img = img.transpose(2, 0, 1)  # to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
        pred = yolov7(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(
        pred,
        conf_thres,
        iou_thres,
        classes=None,
        agnostic=False,
    )

    bboxes = []
    # Process detections
    for det in pred:  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(
                img.shape[2:],
                det[:, :4],
                img0.shape,
            ).round()

            det = det.numpy().astype(np.int32)
            for left, top, right, bottom, conf, cls in det:
                bboxes.append([top, left, bottom, right])

    return np.array(bboxes)


def get_signboard_imgs(image, bboxes):
    signboard_imgs = []
    for bbox in bboxes:
        top, left, bottom, right = bbox
        signboard_imgs.append(image[top:bottom, left:right])
    return signboard_imgs


def run_paddle(signboard_img):
    results = paddle_ocr.ocr(
        signboard_img,
        cls=True,
        # rec=False,
    )
    result = results[0]
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    # scores = [line[1][1] for line in result]
    return boxes, txts


def get_line_images(image, boxes):
    image = np.asarray(image)
    line_imgs = []
    for box in boxes:
        top = int(min(box[0][1], box[1][1]))
        left = int(min(box[0][0], box[3][0]))
        bottom = int(max(box[2][1], box[3][1]))
        right = int(max(box[2][0], box[1][0]))
        line_imgs.append(image[top:bottom, left:right])
    return line_imgs


st.title(":rainbow[Signboard OCR]")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)[:, :, ::-1]

    # Now do something with the image! For example, let's display it:
    st.header(":green[Input image]", divider="rainbow")
    st.image(image)

    signboard_bboxes = detect(image)
    if len(signboard_bboxes) >= 1:
        signboard_imgs = get_signboard_imgs(image, signboard_bboxes)

        st.header(":green[Results]", divider="rainbow")
        for i, signboard_img in enumerate(signboard_imgs):
            st.subheader(f":blue[Signboard {i+1}]")
            st.image(signboard_img)

            line_bboxes, line_texts = run_paddle(signboard_img)
            if len(line_bboxes) >= 1:
                line_imgs = get_line_images(signboard_img, line_bboxes)
                st.subheader(":blue[Line images]")
                for i, line_img in enumerate(line_imgs):
                    st.image(line_img)
                    line_img = Image.fromarray(line_img)
                    s = detector.predict(line_img)
                    Y_pred = gru_model.predict(
                        text_vec_layer([s, line_texts[i]]), verbose=False
                    ).argmax(axis=1)
                    st.text(f"vietocr: {s}, label: {labels[Y_pred[0]]}")
                    st.text(f"paddle: {line_texts[i]}, label: {labels[Y_pred[1]]}")

            else:
                st.text("No texts found")
            st.divider()
    else:
        st.text("No signboards found")

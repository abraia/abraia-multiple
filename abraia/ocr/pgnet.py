# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
    
import cv2
import numpy as np
import onnxruntime

from ..utils import download_file
from .extract_textpoint_fast import generate_pivot_list_fast, restore_poly


def get_dict(character_dict_path):
    character_str = ""
    with open(character_dict_path, "rb") as fin:
        lines = fin.readlines()
        for line in lines:
            character_str += line.decode("utf-8").strip("\n").strip("\r\n")
        dict_character = list(character_str)
    return dict_character


def resize_image_for_totaltext(im, max_side_len=512):
    h, w, _ = im.shape
    resize_w, resize_h = w, h
    ratio = 1.25
    if h * ratio > max_side_len:
        ratio = float(max_side_len) / resize_h
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    max_stride = 128
    resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
    resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
    im = cv2.resize(im, (int(resize_w), int(resize_h)))
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)
    return im, (ratio_h, ratio_w)


def resize_image(im, max_side_len=512):
    """
    resize image to a size multiple of max_stride which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    """
    h, w = im.shape[:2]

    # Fix the longer side
    ratio = max_side_len / max(h, w)
    resize_w = int(resize_w * ratio)
    resize_h = int(resize_h * ratio)

    max_stride = 128
    resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
    resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
    im = cv2.resize(im, (int(resize_w), int(resize_h)))
    ratio_h, ratio_w = resize_h / h, resize_w / w

    return im, (ratio_h, ratio_w)


def pg_postprocess_fast(outs_dict, shape_ratio, image_size, lexicon_table, valid_set, score_thresh):
    p_score, p_border = outs_dict["f_score"][0], outs_dict["f_border"][0]
    p_char, p_direction = outs_dict["f_char"][0], outs_dict["f_direction"][0]
    instance_yxs_list, seq_strs = generate_pivot_list_fast(p_score, p_char, p_direction, lexicon_table, score_thresh)
    poly_list, keep_str_list = restore_poly(instance_yxs_list, seq_strs, p_border, shape_ratio, image_size, valid_set)
    return poly_list, keep_str_list


class PGNetPredictor:
    def __init__(self):
        dict_src = download_file('multiple/models/ic15_dict.txt')
        self.lexicon_table = get_dict(dict_src)

        providers = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]

        model_src = download_file('multiple/models/pgnet.onnx')
        self.session = onnxruntime.InferenceSession(model_src, providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]

    def preprocess(self, img, max_side_len=768, valid_set="totaltext"):
        if valid_set == "totaltext":
            img, (ratio_h, ratio_w) = resize_image_for_totaltext(img, max_side_len)
        else:
            img, (ratio_h, ratio_w) = resize_image(img, max_side_len)
        img = (img / 255 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = img.astype(np.float32).transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img, (ratio_h, ratio_w)

    def predict(self, img):
        inputs = {self.input_name: img}
        outputs = self.session.run(None, inputs)
        return {"f_border": outputs[0], "f_char": outputs[1], "f_direction": outputs[2], "f_score": outputs[3]}

    def clip_det_res(self, points, image_size):
        img_width, img_height = image_size
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def postprocess(self, preds, shape_ratio, image_size):
        points, texts = pg_postprocess_fast(preds, shape_ratio, image_size, self.lexicon_table, valid_set="totaltext", score_thresh=0.5)
        dt_boxes = np.array([self.clip_det_res(box, image_size) for box in points])
        return dt_boxes, texts

    def __call__(self, img):
        h, w = img.shape[:2]
        img, shape_ratio = self.preprocess(img)
        preds = self.predict(img)
        dt_boxes, strs = self.postprocess(preds, shape_ratio, (w, h))
        return dt_boxes, strs

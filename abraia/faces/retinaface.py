import cv2
import math
import numpy as np
import onnxruntime as ort

from itertools import product as product
from math import ceil

from ..utils import download_file
from ..ops import non_maximum_suppression


def prior_box(min_sizes, steps, image_size):
    feature_maps = [[ceil(image_size[0] / step), ceil(image_size[1] / step)] for step in steps]
    anchors = []
    for k, f in enumerate(feature_maps):
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes[k]:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                dense_cx = [x * steps[k] / image_size[1] for x in [j + 0.5]]
                dense_cy = [y * steps[k] / image_size[0] for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors += [cx, cy, s_kx, s_ky]
    output = np.array(anchors).reshape(-1, 4)
    return output


def generate_anchors(baseSize, ratios, scales):
    anchors = []
    cx, cy = [baseSize * 0.5, baseSize * 0.5]
    for ratio in ratios:
        rW = round(baseSize / math.sqrt(ratio))
        rH = round(rW * ratio)
        for scale in scales:
            [rsW, rsH] = [rW * scale * 0.5, rH * scale * 0.5]
            anchors.append([cx - rsW, cy - rsH, cx + rsW, cy + rsH])
    return anchors


def generate_proposals(anchors, stride, score, bbox, landmark, prob_threshold, scale):
    faces = []
    _, h, w = bbox.shape
    for q, anchor in enumerate(anchors):
        for i in range(h):
            for j in range(w):
                prob = float(score[q + len(anchors), i, j])
                if prob >= prob_threshold:
                    anchorXY = np.array([anchor[0] + j * stride, anchor[1] + i * stride])
                    anchorWH = np.array([anchor[2] - anchor[0], anchor[3] - anchor[1]])
                    cxy = anchorXY + anchorWH * 0.5
                    dx, dy, dw, dh = bbox[4*q:4*(q+1), i, j]
                    wh = anchorWH * np.exp(np.array([dw, dh]))
                    xy = cxy + anchorWH * np.array([dx, dy]) - 0.5 * wh
                    landmarks = landmark[10*q:10*(q+1), i, j].reshape(-1, 2)
                    landmarks = landmarks * (anchorWH * scale + 1) + cxy
                    obj = { 'confidence': prob, 'box': [float(xy[0]), float(xy[1]), float(wh[0]), float(wh[1])], 'keypoints': landmarks }
                    faces.append(obj)
    return faces


def process_stride(results, prob_threshold, stride, scales, scale):
    score = np.squeeze(results[f"face_rpn_cls_prob_reshape_stride{stride}"])
    bbox = np.squeeze(results[f"face_rpn_bbox_pred_stride{stride}"])
    landmark = np.squeeze(results[f"face_rpn_landmark_pred_stride{stride}"])
    anchors = generate_anchors(16, [1], scales)
    return generate_proposals(anchors, stride, score, bbox, landmark, prob_threshold, scale)


class Retinaface:
    def __init__(self):
        self.image_size = (640, 640)
        self.landmarksScale = 0.18181818
        model_src = download_file('multiple/models/retinaface_mnet25_v2.simplified.onnx')
        self.session = ort.InferenceSession(model_src)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]

    def preprocess(self, img):
        height, width = img.shape[:2]
        scale = min(self.image_size[0] / width, self.image_size[1] / height)
        size = (round(width * scale), round(height * scale))
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        left, top, right, bottom = 0, 0, self.image_size[0] - size[0], self.image_size[1] - size[1]
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        img = img.astype(np.float32).transpose((2, 0, 1))
        return np.expand_dims(img, axis=0), scale

    def predict(self, data):
        inputs = {self.input_name: data}
        outputs = self.session.run(self.output_names, inputs)
        return {name: output for name, output in zip(self.output_names, outputs)}

    def detect_faces(self, img, prob_threshold = 0.75, iou_threshold = 0.5):
        data, scale = self.preprocess(img)
        results = self.predict(data)
        face_proposals = []
        face_proposals.extend(process_stride(results, prob_threshold, 32, [32, 16], self.landmarksScale))
        face_proposals.extend(process_stride(results, prob_threshold, 16, [8, 4], self.landmarksScale))
        face_proposals.extend(process_stride(results, prob_threshold, 8, [2, 1], self.landmarksScale))
        faces = non_maximum_suppression(face_proposals, iou_threshold)
        for k, face in enumerate(faces):
            confidence, (x, y, w, h), keypoints = face['confidence'], face['box'], face['keypoints']
            faces[k] = {'confidence': confidence, 'box': [round(x / scale), round(y / scale), round(w / scale), round(h / scale)], 'keypoints': keypoints / scale }
        return faces

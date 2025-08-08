import cv2
import math
import numpy as np
import onnxruntime as ort

from itertools import product as product
from math import ceil

from ..utils import download_file
from .ops import non_maximum_suppression, softmax, search_vector


REFERENCE_FACIAL_POINTS = [[38.2946, 51.6963],
                           [73.5318, 51.5014],
                           [56.0252, 71.7366],
                           [41.5493, 92.3655],
                           [70.7299, 92.2041]]


ref_pts = np.array(REFERENCE_FACIAL_POINTS, dtype=np.float32)


def align_face(img, src_pts, size):
    dst_pts = ref_pts * size / 112 if size != 112 else ref_pts
    src_tri = np.array([src_pts[0], src_pts[1], (src_pts[3] + src_pts[4]) / 2]).astype(np.float32)
    dst_tri = np.array([dst_pts[0], dst_pts[1], (dst_pts[3] + dst_pts[4]) / 2]).astype(np.float32)
    M = cv2.getAffineTransform(src_tri, dst_tri)
    # M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    return cv2.warpAffine(img, M, (size, size), borderValue=0.0)


def find_pose(points):
    """Returns the roll, yaw and pitch angles of the face."""
    LMx = points[:, 0] # horizontal coordinates of landmarks
    LMy = points[:, 1] # vertical coordinates of landmarks
    dPx_eyes = max((LMx[1] - LMx[0]), 1)
    dPy_eyes = LMy[1] - LMy[0]
    angle = np.arctan(dPy_eyes / dPx_eyes) # angle for rotation based on slope
    alpha, beta = np.cos(angle), np.sin(angle)
    # rotated landmarks
    LMxr = (alpha * LMx + beta * LMy + (1 - alpha) * LMx[2] / 2 - beta * LMy[2] / 2) 
    LMyr = (-beta * LMx + alpha * LMy + beta * LMx[2] / 2 + (1 - alpha) * LMy[2] / 2)
    # average distance between eyes and mouth
    dXtot = (LMxr[1] - LMxr[0] + LMxr[4] - LMxr[3]) / 2
    dYtot = (LMyr[3] - LMyr[0] + LMyr[4] - LMyr[1]) / 2
    # average distance between nose and eyes
    dXnose = (LMxr[1] - LMxr[2] + LMxr[4] - LMxr[2]) / 2
    dYnose = (LMyr[3] - LMyr[2] + LMyr[4] - LMyr[2]) / 2
    # relative rotation 0 degree is frontal 90 degree is profile
    Xfrontal = (-90 + 90 / 0.5 * dXnose / dXtot) if dXtot != 0 else 0
    Yfrontal = (-90 + 90 / 0.5 * dYnose / dYtot) if dYtot != 0 else 0
    return round(angle * 180 / np.pi, 2), round(Xfrontal, 2), round(Yfrontal, 2)


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
                    obj = { 'score': prob, 'box': [float(xy[0]), float(xy[1]), float(wh[0]), float(wh[1])], 'keypoints': landmarks }
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
            score, (x, y, w, h), keypoints = face['score'], face['box'], face['keypoints']
            faces[k] = {'score': score, 'box': [round(x / scale), round(y / scale), round(w / scale), round(h / scale)], 'keypoints': keypoints / scale }
        return faces


class FaceAttribute:
    def __init__(self):
        """Age and Gender Prediction"""
        model_src = download_file('multiple/models/faces/genderage.simplified.onnx')
        self.session = ort.InferenceSession(model_src)
        inputs = self.session.get_inputs()
        self.input_size = tuple(inputs[0].shape[2:][::-1])
        self.input_names = [x.name for x in self.session.get_inputs()]
        self.output_names = [x.name for x in self.session.get_outputs()]

    def preprocess(self, transformed_image):
        return cv2.dnn.blobFromImage(transformed_image, 1.0, self.input_size, (0, 0, 0), swapRB=True)

    def postprocess(self, predictions):
        scores = softmax(predictions[:2])
        gender = int(np.argmax(scores))
        age = int(np.round(predictions[2]*100))
        return 'Male' if gender == 1 else 'Female', age, float(scores[gender])

    def predict(self, face):
        blob = self.preprocess(face)
        predictions = self.session.run(self.output_names, {self.input_names[0]: blob})[0][0]
        gender, age, score = self.postprocess(predictions)
        return gender, age, score


class ArcFace:
    def __init__(self):
        model_src = download_file('multiple/models/mobilefacenet-res2-6-10-2-dim512.simplified.onnx')
        self.session = ort.InferenceSession(model_src)
        inputs = self.session.get_inputs()
        self.input_name = inputs[0].name
        self.image_size = tuple(inputs[0].shape[2:])
        self.output_names = [out.name for out in self.session.get_outputs()]

    def calculate_embeddings(self, img):
        blob = cv2.dnn.blobFromImages([img], 1.0, self.image_size, (0, 0, 0), swapRB=True)
        out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return out.flatten()


class FaceRecognizer:
    def __init__(self):
        self.detector = Retinaface()
        self.arcface = ArcFace()

    def detect_faces(self, img):
        return self.detector.detect_faces(img)
    
    def extract_faces(self, img, results=None, size=112):
        results = self.detector.detect_faces(img) if results == None else results
        return [align_face(img, result['keypoints'], size) for result in results]
    
    def represent_faces(self, img, results=None, size=112):
        results = self.detector.detect_faces(img) if results == None else results
        for result in results:
            face = align_face(img, result['keypoints'], size)
            result['embeddings'] = self.arcface.calculate_embeddings(face)
        return results
    
    def identify_faces(self, results, index, threshold=0.45):
        for result in results:
            del result['score']
            result['label'] = 'unknown'
            if len(index):
                idx, scores = search_vector(result['embeddings'], index)
                if scores[idx] > threshold:
                    result['score'] = scores[idx]
                    result['label'] = index[idx]['name']
        return results

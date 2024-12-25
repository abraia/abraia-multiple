
# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import math
import string
# import pyclipper
import numpy as np
import onnxruntime as ort

from ..utils import download_file


def get_char(character_dict_path, use_space_char=False):
    character_str = ""
    with open(character_dict_path, "rb") as fin:
        lines = fin.readlines()
        for line in lines:
            character_str += line.decode("utf-8").strip("\n").strip("\r\n")
    if use_space_char:
        character_str += " "
    return character_str


class BaseRecLabelDecode():
    """ Convert between text-label and text-index """
    def __init__(self, character_dict_path=None, use_space_char=False):
        self.beg_str = "sos"
        self.end_str = "eos"
        character_str = get_char(character_dict_path, use_space_char)
        dict_character = ['blank'] + list(character_str)
        self.dict = {char: i for i, char in enumerate(dict_character)}
        self.character = dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """Convert text-index into text-label."""
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list, conf_list = [], []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                conf_list.append(text_prob[batch_idx][idx] if text_prob is not None else 1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list)))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank

    def __call__(self, preds):
        preds_idx, preds_prob = preds.argmax(axis=2), preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        return text


def makeOffsetPoly(poly, offset):
    ccw, offset = np.sign(offset), np.abs(offset)
    def normalizeVec(vec):
        return vec / np.linalg.norm(vec)
    points = []
    num_points = len(poly)
    for curr in range(num_points):
        prev, next = (curr + num_points - 1) % num_points, (curr + 1) % num_points
        vnnX, vnnY = normalizeVec(poly[next] - poly[curr])
        nnnX, nnnY = np.array([vnnY, -vnnX])
        vpnX, vpnY = normalizeVec(poly[curr] - poly[prev])
        npnX, npnY = np.array([vpnY, -vpnX]) * ccw
        bisn = normalizeVec(np.array([nnnX + npnX, nnnY + npnY]) * ccw)
        bislen = offset /  np.sqrt((1 + nnnX * npnX + nnnY * npnY) / 2)
        points.append(poly[curr] + bisn * bislen)
    return np.array(points).round().astype(np.int32)


class DBPostProcess():
    """The post process for Differentiable Binarization (DB)."""

    def __init__(self, thresh=0.3, box_thresh=0.7, max_candidates=1000, unclip_ratio=2.0):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3

    def unclip(self, box, unclip_ratio):
        area = cv2.contourArea(box)
        perimeter = cv2.arcLength(box, True)
        distance = round(area * unclip_ratio / perimeter)
        # offset = pyclipper.PyclipperOffset()
        # offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        # expanded = np.array(offset.Execute(distance))
        expanded = makeOffsetPoly(box, distance)
        return expanded.reshape(-1, 1, 2)

    def box_score_fast(self, bitmap, _box):
        """Use bbox mean score as the mean score."""
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box = box - np.array([xmin, ymin])
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        index_1, index_4 = (0, 1) if points[1][1] > points[0][1] else (1, 0)
        index_2, index_3 = (2, 3) if points[3][1] > points[2][1] else (3, 2)
        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return np.array(box), min(bounding_box[1])

    def boxes_from_bitmap(self, pred, bitmap, dest_size):
        """_bitmap: single map with shape (1, H, W), whose values are binarized as {0, 1}"""
        dest_width, dest_height = dest_size
        height, width = bitmap.shape

        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_candidates)

        boxes, scores = [], []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            score = self.box_score_fast(pred, points)
            if self.box_thresh > score:
                continue
            box = self.unclip(points, self.unclip_ratio)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box)
            scores.append(score)
        return np.array(boxes, dtype=np.int16), scores

    def __call__(self, outs_dict, image_size):
        pred = outs_dict['maps'][:, 0, :, :]
        segmentation = pred > self.thresh
        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            mask = segmentation[batch_index]
            boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask, image_size)
            boxes_batch.append({'points': boxes})
        return boxes_batch


def resize_img(img, limit_side_len):
    """Resize image to a size multiple of 32 which is required by the network."""
    h, w = img.shape[:2]
    ratio = min(limit_side_len / max(h, w), 1)
    resize_h = round(h * ratio / 32) * 32
    resize_w = round(w * ratio / 32) * 32
    img = cv2.resize(img, (resize_w, resize_h))
    ratio_h, ratio_w = resize_h / h, resize_w / w
    return img, (ratio_h, ratio_w)


def str_count(s):
    """Count the number of Chinese characters. A single English character and a single number
    equal to half the length of Chinese characters."""
    count_zh = count_pu = 0
    s_len = len(s)
    en_dg_count = 0
    for c in s:
        if c in string.ascii_letters or c.isdigit() or c.isspace():
            en_dg_count += 1
        elif c.isalpha():
            count_zh += 1
        else:
            count_pu += 1
    return s_len - math.ceil(en_dg_count / 2)


def sorted_boxes(dt_boxes):
    """Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)
    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
            (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


class TextDetector():
    def __init__(self):
        self.postprocess_op = DBPostProcess(thresh=0.3, box_thresh=0.5, max_candidates=1000, unclip_ratio=1.6)
        det_src = download_file('multiple/models/ocr_det.onnx')
        self.session = ort.InferenceSession(det_src)
        self.input_name = self.session.get_inputs()[0].name
    
    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]
        leftMost, rightMost = xSorted[:2, :], xSorted[2:, :]
        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        tl, bl = leftMost[np.argsort(leftMost[:, 1]), :]
        tr, br = rightMost[np.argsort(rightMost[:, 1]), :]
        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, img_size):
        img_width, img_height = img_size
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        return np.array(dt_boxes_new)

    def preprocess(self, img):
        img, shape_ratio = resize_img(img, limit_side_len=960)
        img = (img / 255 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = img.astype(np.float32).transpose((2, 0, 1))
        return np.expand_dims(img, axis=0)

    def __call__(self, img):
        height, width = img.shape[:2]
        inputs = {self.input_name: self.preprocess(img)}
        outputs = self.session.run(None, inputs)
        post_result = self.postprocess_op({'maps': outputs[0]}, (width, height))
        dt_boxes = post_result[0]['points']
        dt_boxes = self.filter_tag_det_res(dt_boxes, (width, height))
        return dt_boxes


class TextRecognizer():
    def __init__(self):
        self.rec_image_shape = [3, 32, 320]
        self.rec_batch_num = 6
        self.max_text_length = 25
        self.limited_max_width = 1280
        self.limited_min_width = 16

        char_dict_src = download_file('multiple/models/ppocr_keys_v1.txt')
        self.postprocess_op = BaseRecLabelDecode(char_dict_src, use_space_char=True)

        rec_src = download_file('multiple/models/ocr_rec.onnx')
        self.session = ort.InferenceSession(rec_src)
       
    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        max_wh_ratio = max(max_wh_ratio, imgW / imgH)
        imgW = int((32 * max_wh_ratio))
        imgW = max(min(imgW, self.limited_max_width), self.limited_min_width)
        h, w = img.shape[:2]
        ratio_imgH = math.ceil(imgH * w / h)
        ratio_imgH = max(ratio_imgH, self.limited_min_width)
        resized_w = imgW if ratio_imgH > imgW else int(ratio_imgH)
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = (resized_image / 255 - 0.5) / 0.5
        resized_image = resized_image.astype(np.float32).transpose((2, 0, 1))
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list):
        img_num = len(img_list)
        width_list = [img.shape[1] / img.shape[0] for img in img_list]
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))

        rec_res = [['', 0.0]] * img_num
        batch_num = self.rec_batch_num
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                # h, w = img_list[ino].shape[0:2]
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]], max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)

            norm_img_batch = np.concatenate(norm_img_batch)
            ort_inputs = {self.session.get_inputs()[0].name: norm_img_batch}
            ort_outs = self.session.run(None, ort_inputs)
            outputs1 = ort_outs[0]
        
        rec_result = self.postprocess_op(outputs1)
        for rno in range(len(rec_result)):
            rec_res[indices[beg_img_no + rno]] = rec_result[rno]
        return rec_res


class TextSystem():
    def __init__(self):
        self.text_detector = TextDetector()
        self.text_recognizer = TextRecognizer()
        self.drop_score = 0.5
        
    def get_rotate_crop_image(self, img, points):
        width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
        height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(img, M, (width, height), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        dst_img = np.rot90(dst_img) if dst_img_height * 1.0 / dst_img_width >= 1.5 else dst_img
        return dst_img

    def __call__(self, img):
        results = []
        dt_boxes = self.text_detector(img)
        if len(dt_boxes):
            dt_boxes = sorted_boxes(dt_boxes)
            img_crop_list = [self.get_rotate_crop_image(img, tmp_box) for tmp_box in dt_boxes]
            rec_res = self.text_recognizer(img_crop_list)
            for box, (text, score) in zip(dt_boxes, rec_res):
                if score >= self.drop_score:
                    results.append({'box': box.astype(np.int32), 'text': text, 'score': float(score)})
        return results 
        
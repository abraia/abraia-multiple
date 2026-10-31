
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
# PP-OCRv6 Tiny preprocessing and classification defaults follow RapidOCR,
# adapted here to use the SDK's existing ONNX session and cache.

import cv2
import math
import numpy as np

from ...utils.remote import download_file
from ..session import OnnxSessionMixin, ResourceGroup


_PP_OCRV6_TINY_ASSETS = {
    "det": "multiple/models/ocr_v6_tiny_det.onnx",
    "cls": "multiple/models/ocr_cls_ppocrv4_mobile.onnx",
    "rec": "multiple/models/ocr_v6_tiny_rec.onnx",
    "dict": "multiple/models/ocr_v6_tiny_dict.txt",
}


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
            score = float(np.mean(conf_list)) if conf_list else 0.0
            result_list.append((text, score))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank

    def __call__(self, preds):
        preds_idx, preds_prob = preds.argmax(axis=2), preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        return text


def makeOffsetPoly(poly, offset):
    """Offset a convex polygon with round joins using NumPy geometry."""
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    radius = abs(float(offset))
    if len(poly) < 3 or radius == 0:
        return np.round(poly).astype(np.int32)

    orientation = 1 if cv2.contourArea(poly.astype(np.float32), oriented=True) >= 0 else -1
    arc_tolerance = 0.25
    if radius <= arc_tolerance:
        max_step = math.pi / 2
    else:
        max_step = 2 * math.acos(max(-1.0, 1 - arc_tolerance / radius))
    max_step = max(max_step, 1e-3)

    offset_points = []
    for index, point in enumerate(poly):
        previous = poly[index - 1]
        following = poly[(index + 1) % len(poly)]
        incoming = point - previous
        outgoing = following - point
        incoming /= max(np.linalg.norm(incoming), 1e-12)
        outgoing /= max(np.linalg.norm(outgoing), 1e-12)
        incoming_normal = orientation * np.array([incoming[1], -incoming[0]])
        outgoing_normal = orientation * np.array([outgoing[1], -outgoing[0]])

        start = math.atan2(incoming_normal[1], incoming_normal[0])
        end = math.atan2(outgoing_normal[1], outgoing_normal[0])
        if orientation > 0:
            angle = (end - start) % (2 * math.pi)
        else:
            angle = -((start - end) % (2 * math.pi))
        segments = max(1, int(math.ceil(abs(angle) / max_step)))
        for step in range(segments + 1):
            theta = start + angle * step / segments
            offset_points.append(
                point + radius * np.array([math.cos(theta), math.sin(theta)])
            )
    return np.round(offset_points).astype(np.int32)


class DBPostProcess():
    """The post process for Differentiable Binarization (DB)."""

    def __init__(self, thresh=0.3, box_thresh=0.7, max_candidates=1000,
                 unclip_ratio=2.0, use_dilation=False):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.dilation_kernel = (
            np.ones((2, 2), dtype=np.uint8) if use_dilation else None
        )

    def unclip(self, box, unclip_ratio):
        area = cv2.contourArea(box)
        perimeter = cv2.arcLength(box, True)
        if perimeter <= 0:
            return np.asarray(box).reshape(-1, 1, 2)
        distance = area * unclip_ratio / perimeter
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
        if self.dilation_kernel is not None:
            bitmap = cv2.dilate(
                bitmap.astype(np.uint8), self.dilation_kernel
            )
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


def resize_img(img, limit_side_len, limit_type="max"):
    """Resize image to a size multiple of 32 which is required by the network."""
    h, w = img.shape[:2]
    if limit_type == "min":
        ratio = max(limit_side_len / min(h, w), 1)
    else:
        ratio = min(limit_side_len / max(h, w), 1)
    resize_h = round(h * ratio / 32) * 32
    resize_w = round(w * ratio / 32) * 32
    img = cv2.resize(img, (resize_w, resize_h))
    ratio_h, ratio_w = resize_h / h, resize_w / w
    return img, (ratio_h, ratio_w)


def sorted_boxes(dt_boxes):
    """Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    if dt_boxes is None or len(dt_boxes) == 0:
        return []
    y_order = np.argsort(dt_boxes[:, 0, 1], kind="stable")
    boxes = dt_boxes[y_order]
    y_sorted = boxes[:, 0, 1]
    line_ids = np.concatenate([
        [0],
        np.cumsum((np.diff(y_sorted) >= 10).astype(np.int32)),
    ])
    x_sorted = boxes[:, 0, 0]
    return boxes[np.lexsort((x_sorted, line_ids))]


class TextDetector(OnnxSessionMixin):
    def __init__(self, providers=None, accelerator=None):
        self.postprocess_op = DBPostProcess(
            thresh=0.3,
            box_thresh=0.5,
            max_candidates=1000,
            unclip_ratio=1.6,
            use_dilation=True,
        )
        self.limit_side_len = 736
        self.limit_type = "min"
        self.mean = np.array([0.5, 0.5, 0.5])
        self.std = np.array([0.5, 0.5, 0.5])
        det_src = download_file(_PP_OCRV6_TINY_ASSETS["det"])
        self._init_onnx_session(
            det_src,
            providers=providers,
            accelerator=accelerator,
        )
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
        img, shape_ratio = resize_img(
            img,
            limit_side_len=self.limit_side_len,
            limit_type=self.limit_type,
        )
        img = (img.astype(np.float32) / 255 - self.mean) / self.std
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

class TextRecognizer(OnnxSessionMixin):
    def __init__(self, providers=None, accelerator=None):
        self.rec_image_shape = [3, 48, 320]
        self.rec_batch_num = 6

        char_dict_src = download_file(_PP_OCRV6_TINY_ASSETS["dict"])
        self.postprocess_op = BaseRecLabelDecode(char_dict_src, use_space_char=True)

        rec_src = download_file(_PP_OCRV6_TINY_ASSETS["rec"])
        self._init_onnx_session(
            rec_src,
            providers=providers,
            accelerator=accelerator,
        )
       
    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        max_wh_ratio = max(max_wh_ratio, imgW / imgH)
        imgW = int(imgH * max_wh_ratio)
        h, w = img.shape[:2]
        ratio = w / float(h)
        resized_w = min(imgW, int(math.ceil(imgH * ratio)))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = (resized_image / 255 - 0.5) / 0.5
        resized_image = resized_image.astype(np.float32).transpose((2, 0, 1))
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list):
        img_num = len(img_list)
        if not img_num:
            return []
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

class TextClassifier(OnnxSessionMixin):
    """Classify detected text lines as upright or rotated by 180 degrees."""

    def __init__(self, providers=None, accelerator=None, cls_thresh=0.9):
        self.cls_image_shape = [3, 48, 192]
        self.cls_batch_num = 6
        self.cls_thresh = float(cls_thresh)
        self.label_list = ("0", "180")
        cls_src = download_file(_PP_OCRV6_TINY_ASSETS["cls"])
        self._init_onnx_session(
            cls_src,
            providers=providers,
            accelerator=accelerator,
        )
        self.input_name = self.session.get_inputs()[0].name

    def resize_norm_img(self, img):
        img_c, img_h, img_w = self.cls_image_shape
        height, width = img.shape[:2]

        # PP-OCRv4's orientation model classifies the center of long lines.
        keep_width = int(math.ceil(height * img_w / float(img_h)))
        if width > keep_width:
            left = (width - keep_width) // 2
            img = img[:, left:left + keep_width]
            width = keep_width

        ratio = width / float(height)
        resized_width = min(img_w, int(math.ceil(img_h * ratio)))
        resized = cv2.resize(
            img,
            (resized_width, img_h),
            interpolation=cv2.INTER_LINEAR,
        )
        resized = resized.astype(np.float32).transpose((2, 0, 1)) / 255
        resized = (resized - 0.5) / 0.5
        padded = np.zeros((img_c, img_h, img_w), dtype=np.float32)
        padded[:, :, :resized_width] = resized
        return padded

    def __call__(self, img_list):
        if not img_list:
            return []

        img_list = [np.ascontiguousarray(img) for img in img_list]
        widths = [img.shape[1] / float(img.shape[0]) for img in img_list]
        indices = np.argsort(np.asarray(widths))

        for batch_start in range(0, len(img_list), self.cls_batch_num):
            batch_end = min(len(img_list), batch_start + self.cls_batch_num)
            batch_indices = indices[batch_start:batch_end]
            batch = np.stack([
                self.resize_norm_img(img_list[index])
                for index in batch_indices
            ]).astype(np.float32)
            outputs = self.session.run(None, {self.input_name: batch})[0]
            labels = outputs.argmax(axis=1)
            scores = outputs.max(axis=1)

            for index, label_index, score in zip(batch_indices, labels, scores):
                if self.label_list[int(label_index)] == "180" and score > self.cls_thresh:
                    img_list[index] = cv2.rotate(
                        img_list[index], cv2.ROTATE_180
                    )
        return img_list

class TextSystem():
    def __init__(self, drop_score=0.5, model_variant="ppocrv6_tiny",
                 providers=None, accelerator=None):
        if model_variant != "ppocrv6_tiny":
            raise ValueError("Only the 'ppocrv6_tiny' OCR model is supported")
        self.text_detector = None
        self.text_classifier = None
        self.text_recognizer = None
        self._resources = ResourceGroup()
        self.model_variant = model_variant
        self.min_side_len = 30
        self.max_side_len = 2000
        self.min_height = 30
        self.width_height_ratio = 8
        try:
            self.text_detector = self._resources.add(TextDetector(
                providers=providers,
                accelerator=accelerator,
            ))
            self.text_classifier = self._resources.add(TextClassifier(
                providers=providers,
                accelerator=accelerator,
            ))
            self.text_recognizer = self._resources.add(TextRecognizer(
                providers=providers,
                accelerator=accelerator,
            ))
            self.drop_score = float(drop_score)
        except Exception:
            self._resources.close(suppress_errors=True)
            raise

    def get_rotate_crop_image(self, img, points):
        width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
        height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(img, M, (width, height), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        dst_img = np.rot90(dst_img) if dst_img_height * 1.0 / dst_img_width >= 1.5 else dst_img
        return dst_img

    def _prepare_detection_image(self, img):
        """Apply RapidOCR's bounds resize and padding before text detection."""
        original_height, original_width = img.shape[:2]
        height, width = original_height, original_width

        max_side = max(height, width)
        if max_side > self.max_side_len:
            ratio = float(self.max_side_len) / max_side
            resize_height = int(round(int(height * ratio) / 32) * 32)
            resize_width = int(round(int(width * ratio) / 32) * 32)
            img = cv2.resize(img, (resize_width, resize_height))
            height, width = img.shape[:2]

        min_side = min(height, width)
        if min_side < self.min_side_len:
            ratio = float(self.min_side_len) / min_side
            resize_height = int(round(int(height * ratio) / 32) * 32)
            resize_width = int(round(int(width * ratio) / 32) * 32)
            img = cv2.resize(img, (resize_width, resize_height))
            height, width = img.shape[:2]

        ratio_height = original_height / float(height)
        ratio_width = original_width / float(width)
        padding_top = 0
        if height <= self.min_height or width / float(height) > self.width_height_ratio:
            padded_height = max(
                int(width / self.width_height_ratio), self.min_height
            ) * 2
            padding_top = int(abs(padded_height - height) / 2)
            img = cv2.copyMakeBorder(
                img,
                padding_top,
                padding_top,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        return img, ratio_height, ratio_width, padding_top

    def _detect_text_regions(self, img):
        """Detect text and prepare recognition crops for one image."""
        detect_img, ratio_height, ratio_width, padding_top = (
            self._prepare_detection_image(img)
        )
        dt_boxes = self.text_detector(detect_img)
        if len(dt_boxes):
            dt_boxes = dt_boxes.astype(np.float32)
            dt_boxes[:, :, 0] *= ratio_width
            dt_boxes[:, :, 1] = (dt_boxes[:, :, 1] - padding_top) * ratio_height
            dt_boxes[:, :, 0] = np.clip(
                dt_boxes[:, :, 0], 0, img.shape[1] - 1
            )
            dt_boxes[:, :, 1] = np.clip(
                dt_boxes[:, :, 1], 0, img.shape[0] - 1
            )
        dt_boxes = sorted_boxes(dt_boxes)
        return dt_boxes, [
            self.get_rotate_crop_image(img, box) for box in dt_boxes
        ]

    def _recognize_batch(self, images):
        """Recognize text in multiple images with one shared batch stream."""
        image_data = [self._detect_text_regions(img) for img in images]
        crops = [crop for _boxes, image_crops in image_data for crop in image_crops]
        crops = self.text_classifier(crops)
        recognized = self.text_recognizer(crops)
        results = [[] for _ in images]
        offset = 0
        for image_index, (boxes, _image_crops) in enumerate(image_data):
            for box in boxes:
                if offset >= len(recognized):
                    break
                text, score = recognized[offset]
                offset += 1
                if not text.strip() or score < self.drop_score:
                    continue
                results[image_index].append({
                    'box': box.astype(np.int32),
                    'text': text,
                    'score': float(score),
                })
        return results

    def recognize_crop(self, img, padding=0):
        """Recognize a known single-line text crop without running detection."""
        height, width = img.shape[:2]
        if height == 0 or width == 0:
            return []
        recognizer_img = img
        if padding > 0 and height > 2 * padding and width > 2 * padding:
            recognizer_img = img[padding:-padding, padding:-padding]
        crop_height, crop_width = recognizer_img.shape[:2]
        crop = (
            np.ascontiguousarray(np.rot90(recognizer_img))
            if crop_height / crop_width >= 1.5
            else recognizer_img
        )
        crop = self.text_classifier([crop])[0]
        recognized = self.text_recognizer([crop])
        if not recognized:
            return []
        text, score = recognized[0]
        if not text.strip() or score < self.drop_score:
            return []
        return [{
            "box": np.array(
                [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                dtype=np.int32,
            ),
            "text": text,
            "score": float(score),
        }]

    def __call__(self, img):
        return self._recognize_batch([img])[0]

    def run_batch(self, images):
        """Recognize text for multiple region images in shared batches."""
        return [
            self._normalize_results(results)
            for results in self._recognize_batch(list(images))
        ]

    @staticmethod
    def _normalize_results(results):
        normalized = []
        for result in results:
            result = dict(result)
            points = np.asarray(result.get('box'))
            if points.ndim == 2 and points.shape[1] == 2:
                x1, y1 = points.min(axis=0)
                x2, y2 = points.max(axis=0)
                result['points'] = points
                result['box'] = [
                    int(round(x1)),
                    int(round(y1)),
                    int(round(x2 - x1)),
                    int(round(y2 - y1)),
                ]
            result['label'] = result.get('text', '')
            normalized.append(result)
        return normalized

    def run(self, img):
        """Recognize text using the pipeline model protocol."""
        return self.run_batch([img])[0]

    def close(self):
        """Release text detection and recognition sessions."""
        resources, self._resources = getattr(self, "_resources", None), None
        if resources is not None:
            resources.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

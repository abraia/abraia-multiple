import os
import cv2
import math
import numpy as np
import onnxruntime as ort

from .ops import py_cpu_nms, normalize, mask_to_polygon, softmax
from .utils import download_file, load_json, get_color, get_providers
from .utils import load_image, show_image, save_image, Video, render_results


def resize(img, size):
    scale = max(size / img.shape[1], size / img.shape[0])
    width, height = round(scale * img.shape[1]), round(scale * img.shape[0])
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)


def crop(img, size):
    height, width = img.shape[:2]
    left, top = (width - size) // 2, (height - size) // 2
    right, bottom = left + size, top + size
    return img[top:bottom, left:right]


def preprocess(img):
    img = resize(img, 256)
    img = crop(img, 224)
    img = normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return np.expand_dims(img.transpose((2, 0, 1)), axis=0)


def postprocess(outputs, classes):
    probs = softmax(outputs[0].flatten())
    idx = np.argmax(probs)
    return [{'label': classes[idx], 'confidence': probs[idx], 'color': get_color(idx)}]


def scale_size(size, new_size):
    scale = min(new_size[0] / size[0], new_size[1] / size[1])
    return round(scale * size[0]), round(scale * size[1])


def prepare_input(img, shape):
    """Converts the input image array to a (3, height, width) tensor."""
    size = scale_size((img.shape[1], img.shape[0]), (shape[3], shape[2]))
    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    # dw, dh = (shape[3] - size[0]) / 2, (shape[2] - size[1]) / 2
    # left, top, right, bottom = round(dw - 0.1), round(dh - 0.1), round(dw + 0.1), round(dh + 0.1)
    left, top, right, bottom = 0, 0, shape[3] - size[0], shape[2] - size[1]
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img = img.transpose((2, 0, 1)).reshape(shape).astype(np.float32)
    return img / 255


def iou(box1, box2):
    """Calculates the intersection-over-union of two boxes."""
    tl1, wh1, br1 = [box1[0], box1[1]], [box1[2], box1[3]], [box1[0] + box1[2], box1[1] + box1[3]]
    tl2, wh2, br2 = [box2[0], box2[1]], [box2[2], box2[3]], [box2[0] + box2[2], box2[1] + box2[3]]
    intersection_area = np.prod(np.maximum(np.minimum(br1, br2) - np.maximum(tl1, tl2), 0))
    union_area = np.prod(wh1) + np.prod(wh2) - intersection_area;
    return intersection_area / union_area


# def non_maximum_suppression(objects, iou_threshold):
#     results = []
#     objects.sort(key=lambda obj: obj['confidence'], reverse=True)
#     while len(objects) > 0:
#         results.append(objects[0])
#         objects = [obj for obj in objects if iou(obj['box'], objects[0]['box']) < iou_threshold]
    # return results


def non_maximum_suppression(objects, iou_threshold):
    dets = []
    for obj in objects:
        s = obj['confidence']
        x, y, w, h = obj['box']
        dets.append([x, y, x + w, y + h, s])
    if dets:
        idxs = py_cpu_nms(np.array(dets), iou_threshold)
        return [objects[idx] for idx in idxs]
    return []


def sigmoid_mask(z):
    mask = 1 / (1 + np.exp(-z))
    return (255 * mask).astype('uint8')


def get_mask(row, box, size):
    """Extracts the segmentation mask for an object (box) in a row."""
    shape = round(math.sqrt(row.shape[0]))
    mask = row.reshape(shape, shape)
    mask = sigmoid_mask(mask)
    x, y, w, h = box
    mask_x1, mask_y1 = round(x / size[0] * shape), round(y / size[1] * shape)
    mask_x2, mask_y2 = round((x + w) / size[0] * shape), round((y + h) / size[1] * shape)
    mask = mask[mask_y1:mask_y2, mask_x1:mask_x2]
    mask = cv2.resize(mask, (round(w), round(h)), cv2.INTER_LINEAR)
    return mask


def process_output(outputs, size, shape, classes, conf_threshold=0.25, iou_threshold=0.7, approx=0.001):
    """Converts the RAW model output from YOLOv8 to an array of detected
    objects, containing the bounding box, label and the probability.
    """
    output0 = outputs[0][0].astype("float")
    output0 = output0.transpose()
    if len(outputs) == 2:
        output1 = outputs[1][0].astype("float")
        output1 = output1.reshape(output1.shape[0], output1.shape[1] * output1.shape[2])
    img_width, img_height = size
    model_width, model_height = shape[3], shape[2]
    scale = 1 / min(model_width / img_width, model_height / img_height)
    objects = []
    for row in output0:
        xc, yc, w, h = row[:4]
        probs = row[4:4+len(classes)]
        idx = probs.argmax()
        if probs[idx] < conf_threshold:
            continue
        x1, y1 = round((xc - w/2) * scale), round((yc - h/2) * scale)
        x2, y2 = round((xc + w/2) * scale), round((yc + h/2) * scale)
        obj = {'label': classes[idx], 'confidence': probs[idx], 'box': [x1, y1, x2 - x1, y2 - y1], 'color': get_color(idx)}
        if len(outputs) == 2:
            obj['mask'] = row[4+len(classes):]
        objects.append(obj)
    results = non_maximum_suppression(objects, iou_threshold)
    for result in results:
        if len(outputs) == 2:
            x, y, w, h = result['box']
            mask = result['mask'] @ output1
            size = (round(model_width * scale), round(model_height * scale))
            mask = get_mask(mask, result['box'], size)
            mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
            result['polygon'] = mask_to_polygon(mask, (x, y), approx)
            result.pop('mask', None)
            # result['mask'] = mask
    return results


def count_objects(results):
    counts = {}
    colors = {}
    for result in results:
        label, color = result['label'], result['color']
        counts[label] = counts.get(label, 0) + 1
        colors[label] = color
    objects = [{'label': label, 'count': counts[label], 'color': colors[label]} for label in counts.keys()]
    return objects


class Model:
    def load(self, model_uri):
        config_uri = f"{os.path.splitext(model_uri)[0]}.json"
        self.config = load_json(download_file(config_uri))
        self.session = ort.InferenceSession(download_file(model_uri), providers=get_providers())
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.config['inputShape']

    def run(self, img, conf_threshold=0.25, iou_threshold=0.7, approx=0.001):
        if self.config.get('task'):
            img_size = img.shape[1], img.shape[0]
            inputs = {self.input_name: prepare_input(img, self.input_shape)}
            outputs = self.session.run(None, inputs)
            return process_output(outputs, img_size, self.input_shape, self.config['classes'], conf_threshold, iou_threshold, approx)
        outputs = self.session.run(None, {self.input_name: preprocess(img)})
        return postprocess(outputs, self.config['classes'])
    

def load_model(model_uri):
    model = Model()
    model.load(model_uri)
    return model

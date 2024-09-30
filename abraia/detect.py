import os
import cv2
import math
import numpy as np
import onnxruntime as ort

from PIL import Image

from .utils import download_file, load_json, load_image, get_color, hex_to_rgb
from .video import Video
from . import utils
from . import draw


def resize(img, size):
    scale = max(size / img.shape[1], size / img.shape[0])
    width, height = round(scale * img.shape[1]), round(scale * img.shape[0])
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)


def crop(img, size):
    height, width = img.shape[:2]
    left, top = (width - size) // 2, (height - size) // 2
    right, bottom = left + size, top + size
    return img[top:bottom, left:right]


def normalize(img, mean, std):
    img = (img / 255 - np.array(mean)) / np.array(std)
    return img.astype(np.float32)


def preprocess(img):
    img = np.array(img.convert('RGB'))
    img = resize(img, 256)
    img = crop(img, 224)
    img = normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return np.expand_dims(img.transpose((2, 0, 1)), axis=0)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


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
    left, top, right, bottom = 0, 0, shape[3] - size[0], shape[2] - size[1]
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return (img / 255).astype(np.float32).transpose(2, 0, 1).reshape(shape)


def intersection(box1, box2):
    """Calculates the intersection area of two boxes."""
    box1_x1, box1_y1, box1_x2, box1_y2 = box1
    box2_x1, box2_y1, box2_x2, box2_y2 = box2
    x1, y1 = max(box1_x1, box2_x1), max(box1_y1, box2_y1)
    x2, y2 = min(box1_x2, box2_x2), min(box1_y2, box2_y2)
    return (x2 - x1) * (y2 - y1)


def union(box1, box2):
    """Calculates the union area of two boxes."""
    box1_x1, box1_y1, box1_x2, box1_y2 = box1
    box2_x1, box2_y1, box2_x2, box2_y2 = box2
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    return box1_area + box2_area - intersection(box1, box2)


def iou(box1, box2):
    """Calculates the intersection-over-union of two boxes."""
    box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
    box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]
    return intersection(box1, box2) / (union(box1, box2) + 0.000001)


def non_maximum_suppression(objects, iou_threshold):
    results = []
    objects.sort(key=lambda x: x['confidence'], reverse=True)
    while len(objects) > 0:
        results.append(objects[0])
        objects = [obj for obj in objects if iou(obj['box'], objects[0]['box']) < iou_threshold]
    return results


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
    img_mask = Image.fromarray(mask, "L")
    img_mask = img_mask.resize((round(w), round(h)), Image.BILINEAR)
    mask = np.array(img_mask)
    return mask


def mask_to_polygon(mask, origin):
    """Returns the largest bounding polygon based on the segmentation mask."""
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    polygon = []
    for contour in contours:
        contour = contour.reshape(-1, 2)
        polygon = list(contour) if contour.shape[0] > len(polygon) else polygon
    polygon = [(int(origin[0] + point[0]), int(origin[1] + point[1])) for point in polygon]
    return polygon


def process_output(outputs, size, shape, classes, confidence, iou_threshold):
    """Converts the RAW model output from YOLOv8 to an array of detected
    objects, containing the bounding box, label and the probability.
    """
    img_width, img_height = size
    model_width, model_height = shape[3], shape[2]
    scale = 1 / min(model_width / img_width, model_height / img_height)
    scale_width, scale_height = scale, scale
    output0 = outputs[0][0].astype("float")
    output0 = output0.transpose()
    if len(outputs) == 2:
        output1 = outputs[1][0].astype("float")
        output1 = output1.reshape(output1.shape[0], output1.shape[1] * output1.shape[2])
    objects = []
    for row in output0:
        xc, yc, w, h = row[:4]
        probs = row[4:4+len(classes)]
        idx = probs.argmax()
        if probs[idx] < confidence:
            continue
        x1, y1 = round((xc - w/2) * scale_width), round((yc - h/2) * scale_height)
        x2, y2 = round((xc + w/2) * scale_width), round((yc + h/2) * scale_height)
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
            result['polygon'] = mask_to_polygon(mask, (x, y))
            result.pop('mask', None)
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
        config_src = download_file(config_uri)
        self.config = load_json(config_src)
        model_src = download_file(model_uri)
        self.session = ort.InferenceSession(model_src, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.config['inputShape']

    def run(self, img, confidence=0.25, iou_threshold=0.7):
        if self.config.get('task'):
            img = img if isinstance(img, np.ndarray) else np.array(img.convert("RGB"))
            img_size = img.shape[1], img.shape[0]
            outputs = self.session.run(None, {self.input_name: prepare_input(img, self.input_shape)})
            return process_output(outputs, img_size, self.input_shape, self.config['classes'], confidence, iou_threshold)
        img = Image.fromarray(img) if isinstance(img, np.ndarray) else img
        outputs = self.session.run(None, {self.input_name: preprocess(img)})
        results = postprocess(outputs, self.config['classes'])
        return results
    

def load_model(model_uri):
    model = Model()
    model.load(model_uri)
    return model
    

def render_results(img, results):
    if isinstance(img, np.ndarray):
        return draw.render_results(img, results)
    return utils.render_results(img, results)
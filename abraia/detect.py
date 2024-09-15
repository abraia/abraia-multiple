import os
import re
import cv2
import json
import math
import requests
import tempfile
import numpy as np
import onnxruntime as ort

from PIL import Image, ImageDraw, ImageFont


tempdir = tempfile.gettempdir()

def download_file(url):
    dest = os.path.join(tempdir, os.path.basename(url))
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        r = requests.get(url, allow_redirects=True)
        open(dest, 'wb').write(r.content)
    return dest


def load_json(src):
    with open(src, 'r') as file:
        return json.load(file)


def load_image(src):
    return Image.open(src)


def load_video(src=0, callback=None, output=None):
    cap = cv2.VideoCapture(src)
    if cap.isOpened() == False:
        print("Error opening video file")
        return
    if output:
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(w, h, fps, cap.isOpened())
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (int(w),int(h)))
    win_name = 'Video'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            if callback:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                img = callback(img)
                frame = np.array(img)[:, :, ::-1].copy()
            if output:
                out.write(frame)
            cv2.imshow(win_name, frame)
            if (cv2.waitKey(25) & 0xFF == ord('q')) or cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            # if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    if output:
        out.release()
    cv2.destroyWindow(win_name)


def get_color(idx):
    colors = ['#D0021B', '#F5A623', '#F8E71C', '#8B572A', '#7ED321',
    '#417505', '#BD10E0', '#9013FE', '#4A90E2', '#50E3C2', '#B8E986',
    '#000000', '#545454', '#737373', '#A6A6A6', '#D9D9D9', '#FFFFFF']
    return colors[idx % (len(colors) - 1)]


def hex_to_rgb(hex):
    h = hex.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def resize(img, size):
    width = size if img.height > img.width else round(size * img.width / img.height)
    height = round(size * img.height / img.width) if img.height > img.width else size
    return img.resize((width, height))


def crop(img, size):
    left, top = (img.width - size) // 2, (img.height - size) // 2
    right, bottom = left + size, top + size
    return img.crop((left, top, right, bottom))


def normalize(img, mean, std):
    img = (np.array(img) / 255. - np.array(mean)) / np.array(std)
    return img.astype(np.float32)


def preprocess(img):
    img = img.convert('RGB')
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
    results = [{'label': classes[idx], 'confidence': probs[idx], 'color': get_color(idx)}]
    return results


def prepare_input(img, shape):
    """Converts the input image to RGB an return a (3, height, width) tensor."""
    img = img.convert("RGB")
    img = img.resize((shape[3], shape[2]))
    input = np.array(img) / 255.0
    input = input.transpose(2, 0, 1)
    input = input.reshape(shape)
    return input.astype(np.float32)


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
    """Calculates "Intersection-over-union" coefficient for specified two boxes."""
    return intersection(box1, box2) / union(box1, box2)


def sigmoid_mask(z):
    mask = 1 / (1 + np.exp(-z))
    return 255 * (mask > 0.5).astype('uint8')


def get_mask(row, box, img_width, img_height):
    """Extracts the segmentation mask for an object (box) in a row."""
    size = round(math.sqrt(row.shape[0]))
    mask = row.reshape(size, size)
    mask = sigmoid_mask(mask)
    x1, y1, x2, y2 = box
    mask_x1, mask_y1 = round(x1 / img_width * size), round(y1 / img_height * size)
    mask_x2, mask_y2 = round(x2 / img_width * size), round(y2 / img_height * size)
    mask = mask[mask_y1:mask_y2, mask_x1:mask_x2]
    img_mask = Image.fromarray(mask, "L")
    img_mask = img_mask.resize((round(x2-x1), round(y2-y1)))
    mask = np.array(img_mask)
    return mask


def get_polygon(mask, origin):
    """Calculates the bounding polygon based on the segmentation mask."""
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygon = [[int(contour[0][0]), int(contour[0][1])] for contour in contours[0][0]]
    polygon = [(int(origin[0] + point[0]), int(origin[1] + point[1])) for point in polygon]
    return polygon


def process_output(outputs, size, shape, classes, confidence = 0.5, iou_threshold = 0.5):
    """Converts the RAW model output from YOLOv8 to an array of detected
    objects, containing the bounding box, label and the probability.
    """
    img_width, img_height = size
    model_width, model_height = shape[3], shape[2]
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
        x1, y1 = (xc - w/2) / model_width * img_width, (yc - h/2) / model_height * img_height
        x2, y2 = (xc + w/2) / model_width * img_width, (yc + h/2) / model_height * img_height
        obj = {'label': classes[idx], 'confidence': probs[idx], 'box': [x1, y1, x2, y2], 'color': get_color(idx)}
        if len(outputs) == 2:
            obj['mask'] = row[4+len(classes):]
        objects.append(obj)

    objects.sort(key=lambda x: x['confidence'], reverse=True)
    results = []
    while len(objects) > 0:
        results.append(objects[0])
        objects = [obj for obj in objects if iou(obj['box'], objects[0]['box']) < iou_threshold]

    for result in results:
        if len(outputs) == 2:
            mask = result['mask'] @ output1
            mask = get_mask(mask, (x1, y1, x2, y2), img_width, img_height)
            result['polygon'] = get_polygon(mask, (x1, y1))
            result.pop('mask', None)
    return results


def render_results(img, results):
    draw = ImageDraw.Draw(img, "RGBA")
    for result in results:
        x1, y1 = 0, 0
        label = result.get('label')
        prob = result.get('confidence')
        color = hex_to_rgb(result.get('color'))
        if (label):
            if result.get('polygon'):
                draw.polygon(result['polygon'], fill=(color[0], color[1], color[2], 125), outline=color, width=1)
            if result.get('box'):
                [x1, y1, x2, y2] = result['box']
                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
            text = f" {label} {round(100 * prob, 1)}% "
            font = ImageFont.load_default()
            y1 = max(y1 - 11, 0)
            bbox = draw.textbbox((x1, y1), text, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y1), text, font=font)
    return img


class Model:
    def load(self, model_uri):
        config_uri = f"{os.path.splitext(model_uri)[0]}.json"
        config_src = download_file(config_uri)
        self.config = load_json(config_src)
        model_src = download_file(model_uri)
        self.ort_session = ort.InferenceSession(model_src, providers=['CPUExecutionProvider'])

    def run(self, img, confidence = 0.5, iou_threshold = 0.5):
        if self.config.get('task'):
            input = prepare_input(img, self.config['inputShape'])
            outputs = self.ort_session.run(None, {"images": input})
            return process_output(outputs, img.size, self.config['inputShape'], self.config['classes'], confidence, iou_threshold)
        input = preprocess(img)
        outputs = self.ort_session.run(None, {"input": input})
        results = postprocess(outputs, self.config['classes'])
        return results
    

def load_model(model_uri):
    model = Model()
    model.load(model_uri)
    return model


if __name__ == '__main__':
    src = 'images/birds.jpg'
    model_uri = 'https://api.abraia.me/files/multiple/camera/yolov8n.onnx'

    model = load_model(model_uri)

    img = load_image(src).convert('RGB')
    results = model.run(img)
    print(src, results)

    img = render_results(img, results)
    img.show()


    src = 'images/people-walking.mp4'
    
    def callback(img):
        results = model.run(img)
        img = render_results(img, results)
        return img
    
    load_video(src, callback=callback, output='output.mp4')
    
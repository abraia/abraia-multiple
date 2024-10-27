import cv2
import numpy as np
import onnxruntime as ort

from .ocr import TextSystem
from ..utils import download_file


def iou(box1, box2):
    tl1, wh1, br1 = [box1[0], box1[1]], [box1[2], box1[3]], [box1[0] + box1[2], box1[1] + box1[3]]
    tl2, wh2, br2 = [box2[0], box2[1]], [box2[2], box2[3]], [box2[0] + box2[2], box2[1] + box2[3]]
    intersection_area = np.prod(np.maximum(np.minimum(br1, br2) - np.maximum(tl1, tl2), 0))
    union_area = np.prod(wh1) + np.prod(wh2) - intersection_area;
    return intersection_area / union_area


def nms(objects, iou_threshold = 0.5):
    results = []
    objects.sort(key=lambda obj: obj['confidence'], reverse=True)
    for object in objects:
        non_overlap = True
        for result in results:
            if iou(object['box'], result['box']) >= iou_threshold:
                non_overlap = False
                break
        if non_overlap:
            results.append(object)
    return results


class LisencePlateDetector():
    def __init__(self, threshold = 0.5, iou_threshold = 0.1, out_size = 300):
        self.out_size = out_size
        self.threshold = threshold
        self.iou_threshold = iou_threshold

        lpd_src = download_file('multiple/models/lpd.onnx')
        self.session = ort.InferenceSession(lpd_src)
    
    def __call__(self, img, net_stride = 2**4):
        height, width = img.shape[:2]
        factor = min(288 * max(width, height) / min(width, height), 608) / min(width, height)
        w, h = int(factor * width), int(factor * height)
        w = w // net_stride * net_stride
        h = h // net_stride * net_stride

        rimg = cv2.resize(img, (w, h)).astype(np.float32) / 255
        T = np.expand_dims(rimg, axis=0)
		
        input_name = self.session.get_inputs()[0].name
        pred_onnx = self.session.run(None, {input_name: T})[0]
        Y = np.squeeze(pred_onnx)
        
        objects = []
        probs = Y[...,0]
        affines = Y[...,2:]
        side = ((208 + 40) / 2) / net_stride # 7.75
        yy, xx = np.where(probs > self.threshold)
        for y, x in zip(yy, xx):
            prob = probs[y, x]
            A = affines[y, x].reshape(2, 3)
            A[0, 0], A[1, 1] = max(A[0, 0], 0), max(A[1, 1], 0)
            base = np.matrix([[-0.5, -0.5, 1.],
                              [0.5, -0.5, 1.],
                              [0.5, 0.5, 1.],
                              [-0.5, 0.5, 1.]]).T
            pts = (side * np.dot(A, base).T + np.array([x + .5, y + .5])) * net_stride
            pts = np.array(pts / np.array([w, h])) * np.array([width, height])
            pt1, pt2 = pts.min(axis=0), pts.max(axis=0)
            box = [round(pt1[0]), round(pt1[1]), round(pt2[0] - pt1[0]), round(pt2[1] - pt1[1])]
            objects.append({'box': box, 'confidence': float(prob), 'points': pts.astype(np.int32)})
        results = nms(objects, self.iou_threshold)
        return results


def extract_plate(img, points, out_size, offset=30):
    w1, w2 = np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])
    h1, h2 = np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])

    w, h = max(w1, w2), max(h1, h2)
    max_wh = max(w, h)

    w, h = int(w / max_wh * out_size), int(h / max_wh * out_size)
    x1, y1, x2, y2 = offset, offset, w + offset, h + offset
    t_points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    
    size = (w + 2 * offset, h + 2 * offset)
    H, _ = cv2.findHomography(points, t_points)
    return cv2.warpPerspective(img, H, size, borderValue=0.0)


class ALPR():
    def __init__(self):
        self.lp_detector = LisencePlateDetector(threshold=0.85, iou_threshold=0.15)
        self.text_system = TextSystem()
        self.out_size = 300

    def detect(self, img):
        return self.lp_detector(img)
        
    def recognize(self, img, results):
        for result in results:
            points = result['points']
            img_lp = extract_plate(img, points, self.out_size)
            outputs = self.text_system(img_lp)
            result['lines'] = outputs
        return results

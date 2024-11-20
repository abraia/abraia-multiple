import cv2
import numpy as np
import onnxruntime as ort

from PIL import Image

from ..utils import download_file

ort.set_default_logger_severity(3)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

def post_process(mask):
    """
    Post Process the mask for a smooth boundary by applying Morphological Operations
    Research based on paper: https://www.sciencedirect.com/science/article/pii/S2352914821000757
    args:
        mask: Binary Numpy Mask
    """
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT)
    # mask = np.where(mask < 127, 0, 255).astype(np.uint8)
    return mask


def naive_cutout(img, mask):
    empty = Image.new("RGBA", (img.size), 0)
    cutout = Image.composite(img, empty, mask)
    return cutout


class RemoveBG:
    
    def __init__(self):
        self.image_size = (1024, 1024)
        self.input_mean = (0.485, 0.456, 0.406)
        self.providers = ort.get_available_providers()
        # model_src = download_file('multiple/models/editing/isnet-general-use.onnx')
        model_src = download_file('multiple/models/editing/isnet-medium.onnx')
        self.session = ort.InferenceSession(model_src, providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, img):
        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LINEAR)
        img = img / np.max(img) - np.array(self.input_mean)
        # img = img / 255 - np.array(self.input_mean)
        img = img.transpose((2, 0, 1)).astype(np.float32)
        return np.expand_dims(img, axis=0)
    
    def postprocess(self, out, size):
        pred = out.reshape(self.image_size)
        # ma, mi = np.max(pred), np.min(pred)
        # pred = (pred - mi) / (ma - mi)
        mask = (pred * 255).astype(np.uint8)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR)
        return mask

    def predict(self, img):
        h, w = img.shape[:2]
        inputs = {self.input_name: self.preprocess(img)}
        outputs = self.session.run(None, inputs)
        mask = self.postprocess(outputs[0], (w, h))
        return mask

    def remove(self, img, post_process_mask = False):
        mask = self.predict(img)
        if post_process_mask:
            mask = post_process(mask)
        mask[mask < 127] = 0
        img = np.array(naive_cutout(Image.fromarray(img), Image.fromarray(mask)))
        return img

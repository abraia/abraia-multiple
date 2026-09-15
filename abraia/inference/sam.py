import cv2
import json
import numpy as np

from ..utils import download_file, Sketcher
from .ops import mask_to_box
from .session import close_resource, close_session, create_onnx_session


def get_input_points(prompt):
    prompt = json.loads(prompt)
    points, labels = [], []
    for mark in prompt:
        if mark["type"] == "point":
            points.append(mark["data"])
            labels.append(mark["label"])
        elif mark["type"] == "rectangle":
            points.append([mark["data"][0], mark["data"][1]])
            points.append([mark["data"][2], mark["data"][3]])
            labels.append(2)
            labels.append(3)
    return np.array(points), np.array(labels)


class SAM:

    def __init__(self):
        self.image_embedding = None
        self._closed = False
        self.target_size = 1024
        self.input_size = (684, 1024)
        self.encoder = None
        self.decoder = None
        try:
            encoder_src = download_file('multiple/models/mobile_sam.encoder.onnx')
            decoder_src = download_file('multiple/models/mobile_sam.decoder.onnx')
            self.encoder = create_onnx_session(encoder_src)
            self.decoder = create_onnx_session(decoder_src)
        except Exception:
            close_resource(self.encoder)
            close_resource(self.decoder)
            self.encoder = None
            self.decoder = None
            raise

    def encode(self, img):
        if self._closed:
            raise RuntimeError("SAM has already been closed")
        scale_x = self.input_size[1] / img.shape[1]
        scale_y = self.input_size[0] / img.shape[0]
        scale = min(scale_x, scale_y)

        transform_matrix = np.array([[scale, 0, 0],
                                     [0, scale, 0],
                                     [0, 0, 1]])

        size = (self.input_size[1], self.input_size[0])
        img = cv2.warpAffine(img, transform_matrix[:2], size, flags=cv2.INTER_LINEAR)

        encoder_input_name = self.encoder.get_inputs()[0].name
        encoder_inputs = {encoder_input_name: img.astype(np.float32)}
        encoder_output = self.encoder.run(None, encoder_inputs)
        self.image_embedding = encoder_output[0]
        return self.image_embedding

    def decode(self, image_embedding, input_points, input_labels):
        onnx_coord = np.concatenate([input_points, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
        onnx_label = np.concatenate([input_labels, np.array([-1])], axis=0)[None, :].astype(np.float32)
        onnx_coord = np.concatenate([onnx_coord, np.ones((1, onnx_coord.shape[1], 1), dtype=np.float32)], axis=2)
        onnx_coord = onnx_coord[:, :, :2].astype(np.float32)

        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        decoder_inputs = {"image_embeddings": image_embedding,
                          "point_coords": onnx_coord,
                          "point_labels": onnx_label,
                          "mask_input": onnx_mask_input,
                          "has_mask_input": onnx_has_mask_input,
                          "orig_im_size": np.array(self.input_size, dtype=np.float32)}
        masks, _, _ = self.decoder.run(None, decoder_inputs)
        return masks[0]

    def predict(self, img, prompt="[]"):
        if self._closed:
            raise RuntimeError("SAM has already been closed")
        height, width = img.shape[:2]

        scale_x = self.input_size[1] / img.shape[1]
        scale_y = self.input_size[0] / img.shape[0]
        scale = min(scale_x, scale_y)

        transform_matrix = np.array([[scale, 0, 0],
                                     [0, scale, 0],
                                     [0, 0, 1]])

        if self.image_embedding is None:
            self.image_embedding = self.encode(img)

        input_points, input_labels = get_input_points(prompt)
        input_points = input_points * scale
        masks = self.decode(self.image_embedding, input_points, input_labels)

        inv_transform_matrix = np.linalg.inv(transform_matrix)
        mask = np.zeros((height, width), dtype=np.uint8)
        for m in masks:
            m = cv2.warpAffine(m, inv_transform_matrix[:2], (width, height), flags=cv2.INTER_LINEAR)
            mask[m > 0.0] = 255
        return mask

    def close(self):
        """Release the encoder and decoder sessions."""
        if self._closed:
            return
        for name in ("encoder", "decoder"):
            session, setattr_target = getattr(self, name, None), name
            setattr(self, setattr_target, None)
            close_session(session)
        self.image_embedding = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class InteractiveSAM(Sketcher, SAM):

    def __init__(self, img, radius=7):
        SAM.__init__(self)
        self.image_embedding = None
        self.encode(img)
        self.cropped_img = None
        self.selected_box = None
        Sketcher.__init__(self, img, radius=radius)

    def select_object(self):
        """Interactively select an object by clicking on it, returning the cropped image and bounding box."""
        def handle_click(point):
            prompt = json.dumps([{"type": "point", "data": point, "label": 1}])
            mask = self.predict(self.img, prompt=prompt)
            box = mask_to_box(mask)
            if box:
                x, y, w, h = box
                self.selected_box = box
                self.cropped_img = self.img[y:y+h, x:x+w]
                display_img = self.img.copy()
                cv2.rectangle(display_img, (x, y), (x+w, y+h), (255, 0, 0), 3)
                return display_img, mask
            return self.img, None

        self.on_click(handle_click)
        self.run()
        return self.cropped_img, self.selected_box

    def interactive_mask(self, callback=None):
        """Interactively build a mask via clicks, optionally applying a callback (e.g. for inpainting)."""
        def handle_click(point):
            prompt = json.dumps([{"type": "point", "data": point, "label": 1}])
            mask = self.predict(self.img, prompt=prompt)
            self.mask = cv2.bitwise_or(self.dilate(mask), self.mask)
            if callback:
                return callback(self.img, self.mask)
            return self.img, self.mask

        self.on_click(handle_click)
        return self.run()

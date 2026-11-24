import cv2
import json
import numpy as np

from ...media.display import Sketcher
from ...utils.remote import download_file
from ..geometry import fit_image_size, mask_to_box
from ..session import OnnxSessionBundle


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
    return np.array(points).reshape(-1, 2), np.array(labels)


class SAM:

    def __init__(self, providers=None, accelerator=None):
        self.image_embedding = None
        self._encoded_image = None
        self._closed = False
        self.target_size = 1024
        self.input_size = (684, 1024)
        self.encoder = None
        self.decoder = None
        self._session_bundle = None
        try:
            encoder_src = download_file('multiple/models/mobile_sam.encoder.onnx')
            decoder_src = download_file('multiple/models/mobile_sam.decoder.onnx')
            self._session_bundle = OnnxSessionBundle(
                [encoder_src, decoder_src],
                providers=providers,
                accelerator=accelerator,
            )
            self.encoder, self.decoder = self._session_bundle.sessions
            self.execution_providers = self._session_bundle.execution_providers
            self.accelerator = self._session_bundle.accelerator
        except Exception:
            bundle, self._session_bundle = self._session_bundle, None
            if bundle is not None:
                try:
                    bundle.close()
                except Exception:
                    pass
            self.encoder = None
            self.decoder = None
            raise

    def encode(self, img):
        if self._closed:
            raise RuntimeError("SAM has already been closed")
        source = np.array(img, copy=True)
        scale, _, _ = fit_image_size(source.shape[:2], self.input_size)
        transform = np.array([[scale, 0, 0], [0, scale, 0]])
        resized = cv2.warpAffine(source, transform, self.input_size[::-1], flags=cv2.INTER_LINEAR)

        encoder_input_name = self.encoder.get_inputs()[0].name
        encoder_inputs = {encoder_input_name: resized.astype(np.float32)}
        encoder_output = self.encoder.run(None, encoder_inputs)
        self.image_embedding = encoder_output[0]
        self._encoded_image = source
        return self.image_embedding

    def decode(self, image_embedding, input_points, input_labels):
        onnx_coord = np.concatenate([input_points, [[0.0, 0.0]]], axis=0)[None].astype(np.float32)
        onnx_label = np.concatenate([input_labels, np.array([-1])], axis=0)[None, :].astype(np.float32)

        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        decoder_inputs = {"image_embeddings": image_embedding,
                          "point_coords": onnx_coord,
                          "point_labels": onnx_label,
                          "mask_input": onnx_mask_input,
                          "has_mask_input": onnx_has_mask_input,
                          "orig_im_size": np.array(self.input_size, dtype=np.float32)}
        masks, scores, _ = self.decoder.run(None, decoder_inputs)
        return masks[0], scores[0]

    def predict(self, img, prompt="[]"):
        if self._closed:
            raise RuntimeError("SAM has already been closed")
        height, width = img.shape[:2]

        scale, _, _ = fit_image_size((height, width), self.input_size)
        if (
            self.image_embedding is None or self._encoded_image is None
            or self._encoded_image.dtype != img.dtype
            or not np.array_equal(self._encoded_image, img)
        ):
            self.encode(img)

        input_points, input_labels = get_input_points(prompt)
        input_points = input_points * scale
        masks, scores = self.decode(
            self.image_embedding, input_points, input_labels
        )

        inverse = np.array([[1 / scale, 0, 0], [0, 1 / scale, 0]])
        mask = masks[int(np.argmax(scores))]
        mask = cv2.warpAffine(
            mask, inverse, (width, height),
            flags=cv2.INTER_LINEAR,
        )
        return (mask > 0.0).astype(np.uint8) * 255

    def close(self):
        """Release the encoder and decoder sessions."""
        if self._closed:
            return
        bundle = getattr(self, "_session_bundle", None)
        self._session_bundle = None
        self.encoder = None
        self.decoder = None
        self.image_embedding = None
        self._encoded_image = None
        self._closed = True
        if bundle is not None:
            bundle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class InteractiveSAM(Sketcher, SAM):

    def __init__(self, img, radius=7, providers=None, accelerator=None):
        try:
            SAM.__init__(self, providers=providers, accelerator=accelerator)
            self.encode(img)
            self.cropped_img = None
            self.selected_box = None
            Sketcher.__init__(self, img, radius=radius)
        except Exception:
            try:
                self.close()
            except Exception:
                pass
            raise

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

import cv2
import json
import numpy as np
import onnxruntime as ort

from ..utils import download_file


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
        self.target_size = 1024
        self.input_size = (684, 1024)
        sess_options = ort.SessionOptions()
        providers = ort.get_available_providers()
        encoder_src = download_file('multiple/models/mobile_sam.encoder.onnx')
        decoder_src = download_file('multiple/models/mobile_sam.decoder.onnx')
        self.encoder = ort.InferenceSession(encoder_src, providers=providers, sess_options=sess_options)
        self.decoder = ort.InferenceSession(decoder_src, providers=providers, sess_options=sess_options)

    def encode(self, img):
        encoder_input_name = self.encoder.get_inputs()[0].name
        encoder_inputs = {encoder_input_name: img.astype(np.float32)}
        encoder_output = self.encoder.run(None, encoder_inputs)
        image_embedding = encoder_output[0]
        return image_embedding
    
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
        height, width = img.shape[:2]

        scale_x = self.input_size[1] / img.shape[1]
        scale_y = self.input_size[0] / img.shape[0]
        scale = min(scale_x, scale_y)

        transform_matrix = np.array([[scale, 0, 0],
                                     [0, scale, 0],
                                     [0, 0, 1]])

        size = (self.input_size[1], self.input_size[0])
        img = cv2.warpAffine(img, transform_matrix[:2], size, flags=cv2.INTER_LINEAR)
        image_embedding = self.encode(img)

        # embedding = {"image_embedding": image_embedding, 
        #              "original_size": (width, height), 
        #              "transform_matrix": transform_matrix}

        input_points, input_labels = get_input_points(prompt)
        input_points = input_points * scale
        masks = self.decode(image_embedding, input_points, input_labels)

        inv_transform_matrix = np.linalg.inv(transform_matrix)
        mask = np.zeros((height, width, 3), dtype=np.uint8)
        for m in masks:
            m = cv2.warpAffine(m, inv_transform_matrix[:2], (width, height), flags=cv2.INTER_LINEAR)
            mask[m > 0.0] = [255, 255, 255]
        return mask

import cv2
import onnxruntime as ort

from ..utils import download_file


class ArcFace:
    def __init__(self):
        model_src = download_file('multiple/models/mobilefacenet-res2-6-10-2-dim512.simplified.onnx')
        self.session = ort.InferenceSession(model_src, None)
        self.input_mean, self.input_std = 0.0, 1.0
        inputs = self.session.get_inputs()
        self.input_name = inputs[0].name
        self.image_size = tuple(inputs[0].shape[2:])
        self.output_names = [out.name for out in self.session.get_outputs()]

    def calculate_embeddings(self, img):
        blob = cv2.dnn.blobFromImages([img], 1.0 / self.input_std, self.image_size,
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return out.flatten()

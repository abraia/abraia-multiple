import cv2
import onnxruntime as ort

from ..utils import download_file


class ArcFace:
    def __init__(self):
        #model_src = download_file('https://api.abraia.me/files/multiple/models/mobilefacenet.onnx')
        #self.input_mean, self.input_std = 127.5, 127.5
        model_src = download_file('https://api.abraia.me/files/multiple/models/mobilefacenet-res2-6-10-2-dim512.onnx')
        self.input_mean, self.input_std = 0.0, 1.0
        self.session = ort.InferenceSession(model_src, None)

        inputs = self.session.get_inputs()
        self.input_shape = inputs[0].shape
        self.input_name = inputs[0].name
        self.input_size = tuple(self.input_shape[2:])
        outputs = self.session.get_outputs()
        self.output_names = [out.name for out in outputs]

    def calculate_embeddings(self, img):
        blob = cv2.dnn.blobFromImages([img], 1.0 / self.input_std, self.input_size,
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return out.flatten()

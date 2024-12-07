import cv2
import numpy as np
import onnxruntime as ort

from ..utils import download_file
from ..ops import softmax


class Attribute:
    def __init__(self):
        """Age and Gender Prediction"""
        self.input_std = 1.0
        self.input_mean = 0.0

        model_src = download_file('multiple/models/faces/genderage.simplified.onnx')
        self.session = ort.InferenceSession(model_src, None)

        inputs = self.session.get_inputs()
        self.input_size = tuple(inputs[0].shape[2:][::-1])

        self.input_names = [x.name for x in self.session.get_inputs()]
        self.output_names = [x.name for x in self.session.get_outputs()]

    def preprocess(self, transformed_image):
        return cv2.dnn.blobFromImage(transformed_image, 1.0 / self.input_std,
            self.input_size, (self.input_mean, self.input_mean, self.input_mean), swapRB=True)

    def postprocess(self, predictions):
        scores = softmax(predictions[:2])
        gender = int(np.argmax(scores))
        age = int(np.round(predictions[2]*100))
        return 'Male' if gender == 1 else 'Female', age, float(scores[gender])

    def predict(self, face):
        blob = self.preprocess(face)
        predictions = self.session.run(self.output_names, {self.input_names[0]: blob})[0][0]
        gender, age, score = self.postprocess(predictions)
        return gender, age, score

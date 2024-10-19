from .pgnet import PGNetPredictor

class OCR:
    def __init__(self):
        self.predictor = PGNetPredictor()

    def predict(self, img):
        dt_boxes, strs = self.predictor(img)
        return [{'box': box, 'text': text} for box, text in zip(dt_boxes, strs)]

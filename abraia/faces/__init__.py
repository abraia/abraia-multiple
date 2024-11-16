import numpy as np
from .mtcnn import MTCNN
from .retinaface import Retinaface
from .arcface import ArcFace
from .transform import align_face


def euclidean_distance(feat1, feat2):
    return float(np.linalg.norm(feat1 - feat2))


def cosine_similarity(feat1, feat2):
    return float(np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2)))


class Recognition:
    def __init__(self):
        # self.detector = MTCNN()
        self.detector = Retinaface()
        self.arcface = ArcFace()

    def detect_faces(self, img):
        return self.detector.detect_faces(img)
    
    def extract_faces(self, img, results=None, size=112):
        results = self.detector.detect_faces(img) if results == None else results
        return [align_face(img, result['keypoints'], size) for result in results]
    
    def represent_faces(self, img, results=None, size=112):
        results = self.detector.detect_faces(img) if results == None else results
        for result in results:
            face = align_face(img, result['keypoints'], size)
            result['embeddings'] = self.arcface.calculate_embeddings(face)
        return results
    
    def identify_faces(self, results, index, threshold=0.45):
        for result in results:
            sims = [cosine_similarity(result['embeddings'], ind['embeddings']) for ind in index]
            idx = np.argmax(sims)
            if sims[idx] > threshold:
                result['confidence'] = sims[idx]
                result['label'] = index[idx]['name']
        return results


__all__ = [Recognition]
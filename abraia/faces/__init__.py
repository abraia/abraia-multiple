import numpy as np
from .mtcnn import MTCNN
from .arcface import ArcFace
from .transform import align_faces


def euclidean_distance(feat1, feat2):
    return np.sqrt(np.square(feat1 - feat2).sum())


def cosine_similarity(feat1, feat2):
    return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))


class Recognition:
    def __init__(self):
        self.detector = MTCNN()
        self.arcface = ArcFace()

    def detect_faces(self, img):
        return self.detector.detect_faces(img)
    
    def extract_faces(self, img, results=None):
        if results == None:
            results = self.detector.detect_faces(img)
        return align_faces(img, results)
    
    def represent_faces(self, img, results=None):
        if results == None:
            results = self.detector.detect_faces(img)
        faces = align_faces(img, results)
        for result, face in zip(results, faces):
            result['embeddings'] = self.arcface.calculate_embeddings(face)
        return results
    
    def compute_similarity(self, feat1, feat2):
        return float(cosine_similarity(feat1, feat2))


__all__ = [Recognition]
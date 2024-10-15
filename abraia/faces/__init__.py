import numpy as np
from .mtcnn import MTCNN
from .arcface import ArcFace
from .transform import align_faces, align_face


def euclidean_distance(feat1, feat2):
    return float(np.linalg.norm(feat1 - feat2))


def cosine_similarity(feat1, feat2):
    return float(np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2)))


class Recognition:
    def __init__(self):
        self.detector = MTCNN()
        self.arcface = ArcFace()

    def detect_faces(self, img):
        return self.detector.detect_faces(img)
    
    def extract_faces(self, img, results=None):
        results = self.detector.detect_faces(img) if results == None else results
        return align_faces(img, results)
    
    def represent_faces(self, img, results=None):
        results = self.detector.detect_faces(img) if results == None else results
        for result in results:
            face = align_face(img, result['keypoints'])
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
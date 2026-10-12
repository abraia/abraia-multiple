from functools import lru_cache
import os
import cv2
import numpy as np
from PIL import Image

from ..inference.models.faces import Retinaface


def spectral_residual_mono(gray):
    imfft = cv2.dft(gray.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    magnitude, phase = cv2.cartToPolar(imfft[:, :, 0], imfft[:, :, 1])
    log_amplitude = cv2.log(magnitude + 1e-10)
    exp_magnitude = cv2.exp(log_amplitude - cv2.blur(log_amplitude, (3, 3)))
    spectral_residual = np.dstack(
        [exp_magnitude * np.cos(phase), exp_magnitude * np.sin(phase)])
    salmap = cv2.idft(
        spectral_residual, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)**2
    return salmap


def spectral_residual(img):
    """Computes a salience map with the spectral residual method."""
    rimg = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    if img.ndim == 3:
        rimg = cv2.cvtColor(rimg, cv2.COLOR_RGB2LAB)
        ss = [spectral_residual_mono(rimg[:, :, c]) for c in range(3)]
        salmap = np.sum(np.dstack(ss), axis=2)
    else:
        salmap = spectral_residual_mono(rimg)
    salmap = cv2.GaussianBlur(salmap, (11, 11), 2.5, dst=salmap)
    salmap = cv2.resize(salmap, (img.shape[1], img.shape[0]))
    return cv2.normalize(salmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


def draw_ellipses(img, rects, color=255, thickness=-1):
    for x, y, w, h in rects:
        cv2.ellipse(img, (x+(w-1)//2, y+(h-1)//2), ((w-1)//2, (h-1)//2), 0, 0, 360, color, thickness)
    return img


def combine_maps(map1, map2, alpha=0.3):
    cmap = cv2.addWeighted(map1, 1-alpha, map2, alpha, 0)
    return cv2.normalize(cmap, cmap, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


@lru_cache(maxsize=1)
def _center_map():
    path = os.path.join(os.path.dirname(__file__), "center.jpg")
    return cv2.imread(path)[:, :, 0]


def _apply_face_prior(salmap, faces, alpha):
    if len(faces) == 0:
        return salmap
    face_map = draw_ellipses(
        np.zeros(salmap.shape, dtype=np.uint8), faces, 255
    )
    face_map = cv2.GaussianBlur(
        face_map.astype(np.float32) / 255, (249, 249), 30
    )
    face_map = cv2.normalize(
        face_map,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )
    return combine_maps(salmap, face_map, alpha)


def saliency_map(img, faces=None):
    """Combine spectral, face, and center priors into a saliency map."""
    faces = () if faces is None else faces
    salmap = spectral_residual(img)
    salmap = _apply_face_prior(salmap, faces, 0.35)
    size = salmap.shape[1], salmap.shape[0]
    center = cv2.resize(_center_map(), size)
    return combine_maps(salmap, center, 0.05)


def rectangle_union(recta, rectb):
    """Calculates the rectangle which results from the union of two rectangles.
    """
    x, y = min(recta[0], rectb[0]), min(recta[1], rectb[1])
    w = max(recta[0] + recta[2], rectb[0] + rectb[2]) - x
    h = max(recta[1] + recta[3], rectb[1] + rectb[3]) - y
    return x, y, w, h


def rectangle_intersection(recta, rectb):
    """Calculates the rectangle which results from the intersection of two rectangles."""
    x, y = max(recta[0], rectb[0]), max(recta[1], rectb[1])
    w = min(recta[0] + recta[2], rectb[0] + rectb[2]) - x
    h = min(recta[1] + recta[3], rectb[1] + rectb[3]) - y
    return (x, y, w, h) if w > 0 and h > 0 else (0, 0, 0, 0)


def rectangle_scale(rect, scale):
    """Calculates the rectangle scaled by a defined scale factor."""
    return [round(rect[0] * scale[0]), round(rect[1] * scale[1]),
            round(rect[2] * scale[0]), round(rect[3] * scale[1])]


def rectangle_zoom(rect, zoom):
    """Applies a zoom factor to the rectangle."""
    zw, zh = rect[2] / zoom, rect[3] / zoom
    dx, dy = (zw - rect[2]) / 2, (zh - rect[3]) / 2
    return [round(rect[0] - dx), round(rect[1] - dy), round(zw), round(zh)]


def laplacian(img):
    """Calculates the laplacian edges for a given image."""
    img = cv2.GaussianBlur(img, (5, 5), 0)
    lap = cv2.Laplacian(img, cv2.CV_32F)
    mag = cv2.convertScaleAbs(lap)
    mag = cv2.GaussianBlur(mag, (3, 3), 0)
    if mag.ndim == 3:
        mag = np.max(mag, axis=2)
    mag[mag < np.mean(mag)] = 0
    return mag


def max_location(salmap):
    smap = salmap.copy()
    thick = int(0.1 * min(salmap.shape))
    x, y, w, h = (0, 0, salmap.shape[1], salmap.shape[0])
    cv2.rectangle(smap, (int(x), int(y)), (int(x+w-1), int(y+h-1)), 0, thick)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(smap)
    return maxLoc


def resize_image(img, size, filter=Image.Resampling.LANCZOS):
    im = Image.fromarray(img)
    return np.array(im.resize(size, filter))


def image_roi(img, rect):
    x, y, w, h = rect
    return img[y:y+h, x:x+w]


def crop_region(salmap, faces=None):
    if faces is None:
        faces = []
    h, w = salmap.shape
    if len(faces) > 0:
        recti = faces[0]
        for face in faces:
            recti = rectangle_union(recti, face)
    else:
        x, y = max_location(salmap)
        recti = rectangle_zoom((x-1, y-1, 2, 2), 0.2)
    recti = rectangle_zoom(recti, 0.9)
    recto = [0, 0, w, h]
    return recti, recto


def saliency_area(salmap, rect):
    x, y, w, h = rect
    mask = np.zeros(salmap.shape, dtype=bool)
    mask[y:y+h, x:x+w] = True
    return np.sum(salmap[mask])


def content_preservation(salmap, rect):
    sarea = saliency_area(salmap, rect)
    total = np.sum(salmap)
    return sarea / total if total > 0 else 0.0


def boundary_simplicity(energy, rect):
    x, y, w, h = rect
    top = energy[y, x:x+w]
    bottom = energy[y+h-1, x:x+w]
    left = energy[y:y+h, x]
    right = energy[y:y+h, x+w-1]
    boundary = np.hstack([top, bottom, left, right])
    # boundary = boundary[boundary > np.mean(boundary)]
    return np.mean(boundary)


def ratio_size(width, ratio):
    return int(width), int(width / ratio)


def maximum_width(size, ratio):
    return int(min(size[0], ratio * min(size[1], size[0] / ratio)))


def sliding_size(ratio, step, width):
    for stp in range(0, int(0.9 * width), step):
        sw, sh = ratio_size(width - stp, ratio)
        if sh <= 0:
            return
        yield (sw, sh)


def sliding_window(img, step, size, rect):
    ww, wh = size
    rx, ry, rw, rh = rect
    x1, y1, x2, y2 = rx, ry, rx+rw, ry+rh
    for y in range(y1, y2-wh+1, step):
        for x in range(x1, x2-ww+1, step):
            yield (x, y, img[y:y + wh, x:x + ww])


def max_pooling(img):
    w = img.shape[1] >> 1 << 1
    h = img.shape[0] >> 1 << 1
    dstack = np.dstack([img[0:h:2, 0:w:2], img[0:h:2, 1:w:2],
                        img[1:h:2, 0:w:2], img[1:h:2, 1:w:2]])
    return np.max(dstack, axis=2)


def attention_windows(salmap, ratio, recti, recto, step=2):
    xi, yi, wi, hi = recti
    m = maximum_width((recto[2], recto[3]), ratio)
    smean = np.mean(salmap)
    areas = []
    maxmean = 0
    for sw, sh in sliding_size(ratio, step, m):
        recta = rectangle_union([xi+wi-sw, yi+hi-sh, sw, sh], [xi, yi, sw, sh])
        rect = rectangle_intersection(recta, recto)
        rects, means = [], []
        for (x, y, window) in sliding_window(salmap, step, (sw, sh), rect):
            mean = np.mean(window)
            if mean > maxmean:
                if not areas or 0.1 < (np.sum(window > smean) / (sw * sh)) < 0.7:  # negative space
                    rects.append([x, y, sw, sh])
                    means.append(mean)
        if len(means) > 0:
            rects = np.array(rects)
            means = np.array(means)
            maxmean = np.max(means)
            idxs = means > 0.9 * maxmean
            rects = rects[idxs]
            means = means[idxs]
            idxs = np.argsort(means)[::-1]
            areas.extend(rects[idxs])
        if sw < wi or sh < hi:
            break
        break # disable crop zoom
    return np.array(areas)


def best_crop_area(salmap, faces, energy, ratio):
    original_height, original_width = salmap.shape
    recti, recto = crop_region(salmap, faces)
    salmap = max_pooling(salmap)
    energy = max_pooling(energy)
    energy = energy.astype(np.float32) * salmap
    rectm = [1, 1, salmap.shape[1]-2, salmap.shape[0]-2]
    recto = rectangle_scale(recto, (0.5, 0.5))
    recto = rectangle_intersection(rectm, recto)
    recti = rectangle_scale(recti, (0.5, 0.5))
    recti = rectangle_intersection(recto, recti)
    rects = attention_windows(salmap, ratio, recti, recto)
    if len(rects) == 0:
        return [0, 0, original_width, original_height]
    preservation = np.array([content_preservation(salmap, rect) for rect in rects])
    simplicities = np.array([boundary_simplicity(energy, rect) for rect in rects])
    finite = np.isfinite(preservation) & np.isfinite(simplicities)
    if not np.any(finite):
        return [0, 0, original_width, original_height]
    content = preservation >= 0.8 * np.max(preservation[finite])
    candidates = np.flatnonzero(finite & content)
    if len(candidates) == 0:
        candidates = np.flatnonzero(finite)
    idx = candidates[np.argmin(simplicities[candidates])]
    return rectangle_scale(rects[idx], (2, 2))


class Smartcrop:
    def __init__(self):
        self.face_detector = Retinaface()

    def preprocess(self, img):
        height, width = img.shape[:2]
        max_size, min_size = (300, 300), (8, 8)
        max_scale = max(width / max_size[0], height / max_size[1], 1)
        min_scale = min(width / min_size[0], height / min_size[1])
        scale = max_scale if min_scale > max_scale else min_scale
        size = round(width / scale), round(height / scale)
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    def detect(self, img, size):
        h, w = img.shape[:2]
        if (size[0] != w or size[1] != h) and (w > 24 and h > 24):
            rimg = self.preprocess(img)
            rh, rw = rimg.shape[:2]
            energy = laplacian(rimg)
            faces = [
                result["box"] for result in self.face_detector.detect_faces(rimg)
            ]
            salmap = saliency_map(rimg, faces=faces)
            ar = max(size[0] / size[1], 0.027)
            rect = best_crop_area(salmap, faces, energy, ar)
            return rectangle_scale(rect, (w / rw, h / rh))
        return [0, 0, w, h]

    def transform(self, img, size):
        rect = self.detect(img, size)
        img = image_roi(img, rect)
        return resize_image(img, size)

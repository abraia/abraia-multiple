import os
import cv2
import numpy as np
from PIL import Image

from ..faces import Retinaface


WHITE = (255, 255, 255)


class Faces: 
    def __init__(self):
        self.detector = Retinaface()

    def detect(self, img):
        results = self.detector.detect_faces(img)
        return [result['box'] for result in results]


def _spectral_residual(gray):
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
        ss = [_spectral_residual(rimg[:, :, c]) for c in range(3)]
        salmap = np.sum(np.dstack(ss), axis=2)
    else:
        salmap = _spectral_residual(rimg)
    salmap = cv2.GaussianBlur(salmap, (11, 11), 2.5, dst=salmap)
    salmap = cv2.resize(salmap, (img.shape[1], img.shape[0]))
    return cv2.normalize(salmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


def draw_ellipse(img, rect, color=WHITE, thickness=-1):
    x, y, w, h = rect
    cv2.ellipse(img, (x+(w-1)//2, y+(h-1)//2), ((w-1)//2, (h-1)//2),
                0, 0, 360, color, thickness)  # , cv2.LINE_AA)
    return img


def draw_ellipses(img, rects, color=WHITE, thickness=-1):
    for rect in rects:
        img = draw_ellipse(img, rect, color, thickness)
    return img


def combine_maps(map1, map2, alpha=0.3):
    cmap = cv2.addWeighted(map1, 1-alpha, map2, alpha, 0)
    return cv2.normalize(cmap, cmap, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


class Saliency:
    def __init__(self, model=''):
        self.center = cv2.imread(os.path.join(
            os.path.dirname(__file__), 'center.jpg'))[:, :, 0]
        self.model = model

    def predict(self, img, faces=[]):
        salmap = spectral_residual(img)
        salmap = self.faces_map(salmap, faces)
        modmap = self.center_model((salmap.shape[1], salmap.shape[0]))
        salmap = combine_maps(salmap, modmap, 0.05)
        return salmap

    def salmap(self, fixmap):
        salmap = cv2.GaussianBlur(
            fixmap.astype(np.float32) / 255, (249, 249), 30)
        salmap = cv2.normalize(salmap, None, alpha=0, beta=255,
                               norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return salmap

    def faces_map(self, salmap, rects):
        if len(rects):
            facemap = self.salmap(draw_ellipses(
                np.zeros(salmap.shape, dtype=np.uint8), rects, 255))
            salmap = combine_maps(salmap, facemap, 0.35)
        return salmap

    def center_model(self, size):
        return cv2.resize(self.center, size)

    def region(self, salmap, alpha=0):
        msal = round(np.mean(salmap))
        thr = msal + (255 - msal) * alpha
        thr = thr if thr < 255 else 255
        return salmap > thr

    def salient(self, salmap, labeled):
        labels = np.arange(np.max(labeled)) + 1
        meansal = [np.mean(salmap[labeled == lbl]) for lbl in labels]
        idx = 0
        for k in range(len(meansal)-1):
            if meansal[k] > meansal[k+1]:
                break
            idx = idx + 1
        return labeled == idx+1


def rectangle_union(recta, rectb):
    """Calculates the rectangle which results from the union of two rectangles.
    """
    x = min(recta[0], rectb[0])
    y = min(recta[1], rectb[1])
    w = max(recta[0] + recta[2], rectb[0] + rectb[2]) - x
    h = max(recta[1] + recta[3], rectb[1] + rectb[3]) - y
    return x, y, w, h


def rectangle_intersection(recta, rectb):
    """Calculates the rectangle which results from the intersection of two
    rectangles.
    """
    x = max(recta[0], rectb[0])
    y = max(recta[1], rectb[1])
    w = min(recta[0] + recta[2], rectb[0] + rectb[2]) - x
    h = min(recta[1] + recta[3], rectb[1] + rectb[3]) - y
    if w > 0 and h > 0:
        return x, y, w, h
    return 0, 0, 0, 0


def rectangle_scale(rect, scale):
    """Calculates the rectangle scaled by a defined scale factor."""
    return [int(round(rect[0] * scale[0])), int(round(rect[1] * scale[1])),
            int(round(rect[2] * scale[0])), int(round(rect[3] * scale[1]))]


def rectangle_zoom(rect, zoom):
    """Applies a zoom factor to the rectangle."""
    zw, zh = rect[2] / zoom, rect[3] / zoom
    dx, dy = (zw - rect[2]) / 2, (zh - rect[3]) / 2
    return [int(round(rect[0] - dx)), int(round(rect[1] - dy)),
            int(round(zw)), int(round(zh))]


def rectangle_sort(rects):
    """Sorts rectangles from bigger to smaller area."""
    areas = [w * h for x, y, w, h in rects]
    indexes = np.argsort(areas)[::-1]
    return [rects[i] for i in indexes]

# Metrics to compare regions

def overlap_ratio(recta, rectb):
    """Calculates the overlaping ratio between two rectangular regions.

    Arguments:
        recta: Rectangle of reference.
        rectb: Rectangle of comparison.

    Returns:
        The overlap ratio.
    """
    recti = rectangle_intersection(recta, rectb)
    rectu = rectangle_union(recta, rectb)
    return float(recti[2] * recti[3]) / (rectu[2] * rectu[3])


# def boundary_displacement(recta, rectb):
#     """Calculates the boundary displacement error as the average distance
#     between the rectangles sides.

#     Arguments:
#         recta: Rectangle of reference.
#         rectb: Rectangle of comparison.

#     Returns:
#         The boundary displacement error.
#     """
#     dleft = recta[0] - rectb[0]
#     dtop = recta[1] - rectb[1]
#     dright = (recta[0] + recta[2]) - (rectb[0] + rectb[2])
#     dbottom = (recta[1] + recta[3]) - (rectb[1] + rectb[3])
#     return float(dleft + dtop + dright + dbottom) / 4


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


def maximal_size(img, size):
    """Calculates the maximum image size if this is bigger than size."""
    max_scale = max(img.shape[1] / size[0], img.shape[0] / size[1])
    if max_scale > 1:
        return round(img.shape[1] / max_scale), round(img.shape[0] / max_scale)
    return img.shape[1], img.shape[0]


def minimal_size(img, size):
    """Calculates the minimum image size equal or bigger than size."""
    min_scale = min(img.shape[1] / size[0], img.shape[0] / size[1])
    return round(img.shape[1] / min_scale), round(img.shape[0] / min_scale)


def aspect_ratio(size):
    """Calculates the aspect ratio of the size."""
    return float(size[0]) / size[1]


def size_scale(size1, size2):
    """Calculates the scale factor to transform from size1 to size2."""
    return [float(size2[0]) / size1[0],
            float(size2[1]) / size1[1]]


def minimal_rectangle_contains(rect, ar):
    """Calculates the minimal rectangle with a defined aspect retio which
    contains rect and it is center on it.
    """
    xz, yz, wz, hz = rect
    wo, ho = (ar * hz, hz) if (ar * hz) > wz else (wz, wz / ar)
    return [int(round(xz+wz/2-wo/2)), int(round(yz+hz/2-ho/2)),
            int(round(wo)), int(round(ho))]


def rotate(img, angle):
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    return cv2.warpAffine(img, M, (cols, rows))


def normal_resize(img, size, filter=Image.Resampling.LANCZOS):
    im = Image.fromarray(img)
    return np.array(im.resize(size, filter))


def resize(img, size):
    if size[0] < img.shape[1] and size[1] < img.shape[0]:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    else:
        img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    return img


def roi(img, rect):
    x, y, w, h = rect
    return img[y:y+h, x:x+w]


def _preprocess(img):
    max_size = maximal_size(img, (300, 300))
    min_size = minimal_size(img, (8, 8))
    if max_size[0] > min_size[0] and max_size[1] > min_size[1]:
        min_size = max_size
    return cv2.resize(img, min_size, interpolation=cv2.INTER_AREA)


def crop_region(salmap, faces):
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


def best_crop_rect(img, salmap, faces, ar):
    energy = laplacian(img)
    recti, recto = crop_region(salmap, faces)
    rect = best_crop_area(salmap, energy, ar, recti, recto)
    return rect


def saliency_area(salmap, rect):
    x, y, w, h = rect
    mask = np.zeros(salmap.shape, dtype=bool)
    mask[y:y+h, x:x+w] = True
    return np.sum(salmap[mask])


def content_preservation(salmap, rect):
    sarea = saliency_area(salmap, rect)
    return float(sarea) / np.sum(salmap)


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
    return int(width), int(float(width) / ratio)


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
    dstack = np.dstack([img[0:h:2, 0:w:2],
                        img[0:h:2, 1:w:2],
                        img[1:h:2, 0:w:2],
                        img[1:h:2, 1:w:2]])
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
    return np.array(areas)


def best_crop_area(salmap, energy, ratio, recti, recto):
    # salmap = cv2.resize(salmap, None, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    # energy = cv2.resize(energy, None, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    salmap = max_pooling(salmap)
    energy = max_pooling(energy)
    energy = energy.astype(np.float32) * salmap
    rectm = [1, 1, salmap.shape[1]-2, salmap.shape[0]-2]
    recto = rectangle_scale(recto, (0.5, 0.5))
    recto = rectangle_intersection(rectm, recto)
    recti = rectangle_scale(recti, (0.5, 0.5))
    recti = rectangle_intersection(recto, recti)
    rects = attention_windows(salmap, ratio, recti, recto)
    # TODO: Add conditions to attention windows, progressive selection of best cropping areas
    preservation = np.array([
        content_preservation(salmap, rect) for rect in rects])
    simplicities = np.array([
        boundary_simplicity(energy, rect) for rect in rects])
    content = preservation > 0.8 * np.max(preservation)
    if not np.all(content):
        simplicities = simplicities[:np.argmin(content)]
    idx = np.argmin(simplicities)
    return rectangle_scale(rects[idx], (2, 2))


class Transform:
    def __init__(self):
        self.faces = Faces()
        self.saliency = Saliency()

    def get_size(self, img, width, height):
        ar = aspect_ratio([img.shape[1], img.shape[0]])
        if width and not height:
            height = int(round(width / ar))
        if height and not width:
            width = int(round(ar * height))
        width = width or img.shape[1]
        height = height or img.shape[0]
        return width, height

    def smart_crop(self, img, ar):
        h, w = img.shape[:2]
        rimg = _preprocess(img)
        rh, rw = rimg.shape[:2]
        faces = self.faces.detect(rimg)
        salmap = self.saliency.predict(rimg, faces=faces)
        rect = best_crop_rect(rimg, salmap, faces, ar)
        return rectangle_scale(rect, size_scale((rw, rh), (w, h)))

    def apply_transform(self, img, width=None, height=None):
        size = self.get_size(img, width, height)
        if size[0] != img.shape[1] or size[1] != img.shape[0]:
            if img.shape[1] > 24 and img.shape[0] > 24:
                rect = self.smart_crop(img, max(aspect_ratio(size), 0.027))
                img = roi(img, rect)
            return normal_resize(img, size)
        return img

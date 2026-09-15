import cv2
import numpy as np
import onnxruntime as ort

from ..utils import download_file


def _tile_starts(length, tile_length, overlap):
    step = tile_length - overlap
    if tile_length <= 0 or overlap < 0 or overlap >= tile_length:
        raise ValueError('overlap must be non-negative and smaller than the tile dimensions')
    if length <= 0:
        return []
    if length <= tile_length:
        return [0]
    starts = list(range(0, length, step))
    if starts[-1] + tile_length > length:
        starts[-1] = max(0, length - tile_length)
    return list(dict.fromkeys(starts))


def tiling(img, tile_size, overlap = 8):
    """Return an image in a tiled manner."""
    tile_width, tile_height = tile_size
    height, width = img.shape[:2]
    for y in _tile_starts(height, tile_height, overlap):
        y_end = min(y + tile_height, height)
        for x in _tile_starts(width, tile_width, overlap):
            x_end = min(x + tile_width, width)
            tile = img[y:y_end, x:x_end]
            yield tile


def stitching(tiles, img_size, overlap = 8):
    """Return an image from a set of tiles."""
    tiles = list(tiles)
    if not tiles:
        raise ValueError('at least one tile is required')
    tile_height, tile_width = tiles[0].shape[:2]
    overlap = min(overlap, max(tile_height - 1, 0), max(tile_width - 1, 0))
    width, height = img_size
    channels = tiles[0].shape[2] if tiles[0].ndim == 3 else 1
    out = np.zeros((height, width, channels), dtype=np.float32)
    weights = np.zeros((height, width, 1), dtype=np.float32)
    k = 0
    for y in _tile_starts(height, tile_height, overlap):
        for x in _tile_starts(width, tile_width, overlap):
            tile = tiles[k]
            if tile.ndim == 2:
                tile = tile[..., np.newaxis]
            tile_height_actual, tile_width_actual = tile.shape[:2]
            y_end = min(y + tile_height_actual, height)
            x_end = min(x + tile_width_actual, width)
            tile = tile[:y_end - y, :x_end - x]
            out[y:y_end, x:x_end] += tile
            weights[y:y_end, x:x_end] += 1
            k += 1
    if k != len(tiles):
        raise ValueError('tile count does not match image dimensions')
    out = np.divide(out, weights, out=np.zeros_like(out), where=weights != 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def ceil_modulo(x, mod):
    return x if x % mod == 0 else (x // mod + 1) * mod


def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height, out_width = ceil_modulo(height, mod), ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode="symmetric")


class LAMA:
    def __init__(self):
        self.image_size = (512, 512)
        sess_options = ort.SessionOptions()
        model_src = download_file('multiple/models/editing/lama_fp32.onnx')
        self.session = ort.InferenceSession(model_src, sess_options=sess_options)

    def preprocess(self, image, mask, pad_out_to_modulo=8):
        out_image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        out_mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        out_image = (out_image.transpose(2, 0, 1) / 255)
        out_mask = (out_mask[np.newaxis, ...] / 255)

        if pad_out_to_modulo is not None and pad_out_to_modulo > 1:
            out_image = pad_img_to_modulo(out_image, pad_out_to_modulo)
            out_mask = pad_img_to_modulo(out_mask, pad_out_to_modulo)

        out_mask = (out_mask > 0) * 1

        out_image = np.expand_dims(out_image, axis=0).astype(np.float32)
        out_mask = np.expand_dims(out_mask, axis=0).astype(np.float32)
        return out_image, out_mask
    
    def postprocess(self, output, size):
        output = output[0].transpose(1, 2, 0).astype(np.uint8)
        output = cv2.resize(output, size, interpolation=cv2.INTER_CUBIC)
        return output

    def process(self, img, mask):
        h, w = img.shape[:2]
        image, mask = self.preprocess(img, mask)
        outputs = self.session.run(None, {'image': image, 'mask': mask})
        output = self.postprocess(outputs[0], (w, h))
        return output
    
    def inpaint(self, img, mask):
        outs = []
        tile_size = self.image_size
        img_size = (img.shape[1], img.shape[0])
        for im, msk in zip(tiling(img, tile_size), tiling(mask, tile_size)):
            outs.append(self.process(im, msk))
        out = stitching(outs, img_size)
        return out

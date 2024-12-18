import cv2
import numpy as np
import onnxruntime as ort

from ..utils import download_file


def tiling(img, tile_size, overlap = 8):
    """Return an image in a tiled manner."""
    tile_width, tile_height = tile_size
    height, width = img.shape[:2]
    for y in range(0, height, tile_height - overlap):
        y_end = min(y + tile_height, height)
        y = max(0, y_end - tile_height)
        for x in range(0, width, tile_width - overlap):
            x_end = min(x + tile_width, width)
            x = max(0, x_end - tile_width)
            tile = img[y:y_end, x:x_end]
            yield tile


def stitching(tiles, img_size, overlap = 8):
    """Return an image from a set of tiles."""
    tile_height, tile_width = tiles[0].shape[:2]
    width, height = img_size
    out = np.empty((height, width, 3), dtype=np.uint8)
    k = 0
    for y in range(0, height, tile_height - overlap):
        for x in range(0, width, tile_width - overlap):
            x_end = min(x + tile_width, width)
            y_end = min(y + tile_height, height)
            x = max(0, x_end - tile_width)
            y = max(0, y_end - tile_height)
            out[y:y_end, x:x_end, :] = tiles[k]
            k += 1
    return out


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
    
import numpy as np
import onnxruntime as ort

from ..utils import download_file

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL


class SwinIR:

    def __init__(self):
        model_src = download_file('multiple/models/editing/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.onnx')
        self.session = ort.InferenceSession(model_src, sess_options)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, img):
        img = (img / 255).astype(np.float32)
        return np.expand_dims(img.transpose((2, 0, 1)), axis=0)

    def postprocess(self, img):
        img = (255 * img.clip(0, 1)).astype(np.uint8)
        return img.transpose((1, 2, 0))

    def upscale(self, img):
        inputs = {self.input_name: self.preprocess(img)}
        outputs = self.session.run(None, inputs)
        return self.postprocess(outputs[0][0])


def create_gradient_mask(shape, feather):
    """Create a gradient mask for smooth blending of tiles."""
    mask = np.ones(shape)
    _, _, h, w = shape
    for feather_step in range(feather):
        factor = (feather_step + 1) / feather
        mask[:, :, feather_step, :] *= factor
        mask[:, :, h - 1 - feather_step, :] *= factor
        mask[:, :, :, feather_step] *= factor
        mask[:, :, :, w - 1 - feather_step] *= factor
    return mask


def tiled_upscale(samples, function, scale, tile_size, overlap = 8):
    """Apply a scaling function to image samples in a tiled manner."""
    tile_width, tile_height = tile_size
    _batch, _channels, height, width = samples.shape
    out_height, out_width = round(height * scale), round(width * scale)
    # Initialize output tensors
    output = np.empty((1, 3, out_height, out_width))
    out = np.zeros((1, 3, out_height, out_width))
    out_div = np.zeros_like(output)
    # Process the image in tiles
    for y in range(0, height, tile_height - overlap):
        for x in range(0, width, tile_width - overlap):
            # Ensure we don't go out of bounds
            x_end = min(x + tile_width, width)
            y_end = min(y + tile_height, height)
            x = max(0, x_end - tile_width)
            y = max(0, y_end - tile_height)
            # Extract and process the tile
            tile = samples[:, :, y:y_end, x:x_end]
            processed_tile = function(tile)
            # Calculate the position in the output tensor
            out_y, out_x = round(y * scale), round(x * scale)
            out_h, out_w = processed_tile.shape[2:]
            # Create a feathered mask for smooth blending
            mask = create_gradient_mask(processed_tile.shape, overlap * scale)
            # Add the processed tile to the output
            out[:, :, out_y : out_y + out_h, out_x : out_x + out_w] += processed_tile * mask
            out_div[:, :, out_y : out_y + out_h, out_x : out_x + out_w] += mask
    # Normalize the output
    output = out / out_div
    return output


class ESRGAN:

    def __init__(self, overlap = 8):
        self.scale = 2
        self.overlap = overlap
        self.tile_size = (1024, 1024)
        # model_src = download_file('multiple/models/editing/4xNomosWebPhoto_RealPLKSR_fp32_opset17.onnx')
        # model_src = download_file('multiple/models/editing/4xNomosUni_span_multijpg_fp32_opset17.onnx')
        model_src = download_file('multiple/models/editing/2xNomosUni_compact_multijpg_ldl_fp32_opset17.onnx')
        self.session = ort.InferenceSession(model_src, sess_options)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, img):
        img = (img / 255).astype(np.float32)
        return np.expand_dims(img.transpose((2, 0, 1)), axis=0)

    def postprocess(self, img):
        img = (255 * img.clip(0, 1)).astype(np.uint8)
        return img.transpose((1, 2, 0))
    
    def predict(self, img):
        return self.session.run(None, {self.input_name: img})[0]

    def upscale(self, img):
        img = self.preprocess(img)
        outputs = tiled_upscale(img, self.predict, self.scale, self.tile_size, self.overlap)
        return self.postprocess(outputs[0])

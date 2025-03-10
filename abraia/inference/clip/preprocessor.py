import numpy as np
from PIL import Image


class Preprocessor:
    """
    Our approach to the CLIP `preprocess` neural net that does not rely on PyTorch.
    The two approaches fully match.
    """

    # Fixed variables that ensure the correct output shapes and values for the `Model` class.
    CLIP_INPUT_SIZE = 224
    # Normalization constants taken from original CLIP:
    # https://github.com/openai/CLIP/blob/3702849800aa56e2223035bccd1c6ef91c704ca8/clip/clip.py#L85
    NORM_MEAN = np.array([0.48145466, 0.4578275, 0.40821073]).reshape((1, 1, 3))
    NORM_STD = np.array([0.26862954, 0.26130258, 0.27577711]).reshape((1, 1, 3))

    @staticmethod
    def _crop_and_resize(img: np.ndarray) -> np.ndarray:
        """Resize and crop an image to a square, preserving the aspect ratio."""

        h, w = img.shape[0:2]
        target_size = Preprocessor.CLIP_INPUT_SIZE

        # Resize so that the smaller dimension matches the required input size.
        resized_h = target_size if h < w else target_size * h // w
        resized_w = target_size * w // h if h < w else target_size

        img_pil = Image.fromarray(img)
        img_pil = img_pil.resize((resized_w, resized_h), resample=Image.BICUBIC)
        img = np.array(img_pil)

        # Now crop to a square
        y_from, x_from = (resized_h - target_size) // 2, (resized_w - target_size) // 2
        img = img[y_from : y_from + target_size, x_from : x_from + target_size, :]
        return img

    def encode_image(self, img: np.ndarray) -> np.ndarray:
        """Preprocesses the images like CLIP's preprocess() function:
        https://github.com/openai/CLIP/blob/3702849800aa56e2223035bccd1c6ef91c704ca8/clip/clip.py#L79

        Args:
            img: numpy array

        Returns:
            img: numpy image after resizing, center cropping and normalization.
        """

        img = Preprocessor._crop_and_resize(img)
        img = img.astype(np.float32) / 255
        img = np.clip(img, 0, 1)  # In case of rounding errors

        # Normalize channels
        img = (img - Preprocessor.NORM_MEAN) / Preprocessor.NORM_STD

        # Mimic the pytorch tensor format for Model class
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        img = img.astype(np.float32)

        return img

import os
import logging
import numpy as np
import onnxruntime as ort

from PIL import Image
from typing import Union, Iterable, Optional

from .preprocessor import Preprocessor
from .tokenizer import Tokenizer

from ...utils import download_url


class Clip:
    """
    This class can be utilised to predict the most relevant text snippet, given
    an image, without directly optimizing for the task. This class don't depend on
    `torch` or `torchvision`.
    """

    def __init__(self, batch_size: Optional[int] = None, cache_dir: Optional[str] = 'models'):
        """
        Instantiates the model and required encoding classes.

        Args:
            batch_size: If set, splits the lists in `get_image_embeddings`
                and `get_text_embeddings` into batches of this size before
                passing them to the model. The embeddings are then concatenated
                back together before being returned. This is necessary when
                passing large amounts of data (perhaps ~100 or more).
            cache_dir: If provided, the models will be downloaded to / loaded from this location
        """

        IMAGE_MODEL_FILE = "clip_image_model_vitb32.onnx"
        TEXT_MODEL_FILE = "clip_text_model_vitb32.onnx"

        # TODO: Refactor to load from Abraia
        self.image_model = Clip._load_model(os.path.join(cache_dir, IMAGE_MODEL_FILE))
        self.text_model = Clip._load_model(os.path.join(cache_dir, TEXT_MODEL_FILE))
        
        self.embedding_size = 512
        self._tokenizer = Tokenizer()
        self._preprocessor = Preprocessor()
        self._batch_size = batch_size

    @staticmethod
    def _load_model(path: str):
        if not os.path.exists(path):
            s3_url = f"https://lakera-clip.s3.eu-west-1.amazonaws.com/{os.path.basename(path)}"
            logging.info(f"The model file ({path}) doesn't exist or it is invalid. "
                f"Downloading it from the public S3 bucket: {s3_url}.")
            download_url(s3_url, path)
        return ort.InferenceSession(path, providers=ort.get_available_providers())
            
    def get_image_embeddings(self, images: Iterable[Union[Image.Image, np.ndarray]]) -> np.ndarray:
        """Compute the embeddings for a list of images.

        Args:
            images: A list of images to run on. Each image must be a 3-channel
                (RGB) image. Can be any size, as the preprocessing step will
                resize each image to size (224, 224).

        Returns:
            An array of embeddings of shape (len(images), embedding_size).
        """
        embeddings = []
        for batch in to_batches(images, self._batch_size):
            images = [self._preprocessor.encode_image(image) for image in images]
            if not images:
                return self._get_empty_embedding()
            batch = np.concatenate(images)
            embeddings.append(self.image_model.run(None, {"IMAGE": batch})[0])
        if not embeddings:
            return self._get_empty_embedding()
        return np.concatenate(embeddings)

    def get_text_embeddings(self, texts: Iterable[str]) -> np.ndarray:
        """Compute the embeddings for a list of texts.

        Args:
            texts: A list of texts to run on. Each entry can be at most
                77 characters.

        Returns:
            An array of embeddings of shape (len(texts), embedding_size).
        """
        embeddings = []
        for batch in to_batches(texts, self._batch_size):
            text = self._tokenizer.encode_text(batch)
            if len(text) == 0:
                return self._get_empty_embedding()
            embeddings.append(self.text_model.run(None, {"TEXT": text})[0])
        if not embeddings:
            return self._get_empty_embedding()
        return np.concatenate(embeddings)

    def _get_empty_embedding(self):
        return np.empty((0, self.embedding_size), dtype=np.float32)



def to_batches(items, size):
    """
    Splits an iterable (e.g. a list) into batches of length `size`. Includes
    the last, potentially shorter batch.

    Examples:
        >>> list(to_batches([1, 2, 3, 4], size=2))
        [[1, 2], [3, 4]]
        >>> list(to_batches([1, 2, 3, 4, 5], size=2))
        [[1, 2], [3, 4], [5]]

        # To limit the number of batches returned
        # (avoids reading the rest of `items`):
        >>> import itertools
        >>> list(itertools.islice(to_batches([1, 2, 3, 4, 5], size=2), 1))
        [[1, 2]]

    Args:
        items: The iterable to split.
        size: How many elements per batch.
    """
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    # The last, potentially incomplete batch
    if batch:
        yield batch


__all__ = [Clip]

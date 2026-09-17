import os
import logging
import gzip
import html
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Union

import ftfy
import numpy as np
import regex as re

from PIL import Image

from ...utils import download_url, download_file
from ..session import (
    OnnxSessionBundle,
    close_resource,
    close_session,
)


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

        self.image_model = None
        self.text_model = None
        self._session_bundle = None
        try:
            image_model_path = download_file('multiple/models/clip/clip_image_model_vitb32.onnx')
            text_model_path = download_file('multiple/models/clip/clip_text_model_vitb32.onnx')
            image_model_path = Clip._resolve_model_path(image_model_path)
            text_model_path = Clip._resolve_model_path(text_model_path)
            self._session_bundle = OnnxSessionBundle(
                [image_model_path, text_model_path]
            )
            self.image_model, self.text_model = self._session_bundle.sessions
            self.execution_providers = self._session_bundle.execution_providers
            self.accelerator = self._session_bundle.accelerator

            self.embedding_size = 512
            self._tokenizer = Tokenizer()
            self._preprocessor = Preprocessor()
            self._batch_size = batch_size
        except Exception:
            bundle, self._session_bundle = self._session_bundle, None
            if bundle is not None:
                close_resource(bundle)
            else:
                close_resource(self.image_model)
                close_resource(self.text_model)
            self.image_model = None
            self.text_model = None
            raise

    @staticmethod
    def _resolve_model_path(path: str):
        if not os.path.exists(path):
            s3_url = f"https://lakera-clip.s3.eu-west-1.amazonaws.com/{os.path.basename(path)}"
            logging.info(f"The model file ({path}) doesn't exist or it is invalid. "
                f"Downloading it from the public S3 bucket: {s3_url}.")
            download_url(s3_url, path)
        return path

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
            imgs = [self._preprocessor.encode_image(image) for image in batch]
            if not imgs:
                return self._get_empty_embedding()
            batch_arr = np.concatenate(imgs)
            embeddings.append(self.image_model.run(None, {"IMAGE": batch_arr})[0])
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

    def close(self):
        """Release the image and text ONNX sessions."""
        bundle, self._session_bundle = getattr(self, '_session_bundle', None), None
        if bundle is not None:
            bundle.close()
        for name in ("image_model", "text_model"):
            session = getattr(self, name, None)
            setattr(self, name, None)
            if bundle is None:
                close_session(session)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()



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


class Preprocessor:
    """Preprocess images using the original CLIP image transformations."""

    CLIP_INPUT_SIZE = 224
    NORM_MEAN = np.array([0.48145466, 0.4578275, 0.40821073]).reshape((1, 1, 3))
    NORM_STD = np.array([0.26862954, 0.26130258, 0.27577711]).reshape((1, 1, 3))

    @staticmethod
    def _crop_and_resize(img: np.ndarray) -> np.ndarray:
        """Resize and crop an image to a square, preserving its aspect ratio."""
        h, w = img.shape[0:2]
        target_size = Preprocessor.CLIP_INPUT_SIZE
        resized_h = target_size if h < w else target_size * h // w
        resized_w = target_size * w // h if h < w else target_size
        img = Image.fromarray(img).resize(
            (resized_w, resized_h), resample=Image.BICUBIC
        )
        img = np.array(img)
        y_from = (resized_h - target_size) // 2
        x_from = (resized_w - target_size) // 2
        return img[y_from:y_from + target_size, x_from:x_from + target_size, :]

    def encode_image(self, img: np.ndarray) -> np.ndarray:
        """Convert an image to the normalized CLIP input tensor."""
        img = self._crop_and_resize(img)
        img = np.clip(img.astype(np.float32) / 255, 0, 1)
        img = (img - self.NORM_MEAN) / self.NORM_STD
        return np.expand_dims(img.transpose((2, 0, 1)), axis=0).astype(np.float32)


def default_bpe():
    """Return the path to the CLIP vocabulary distributed with Multiple."""
    return str(
        Path(__file__).resolve().parents[3]
        / "multiple"
        / "models"
        / "clip"
        / "bpe_simple_vocab_16e6.txt.gz"
    )


def bytes_to_unicode() -> Dict[int, str]:
    """Create the reversible byte-to-unicode mapping used by CLIP."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, (chr(value) for value in cs)))


def get_pairs(word) -> Set[str]:
    """Return adjacent symbol pairs in a token."""
    pairs = set()
    previous = word[0]
    for character in word[1:]:
        pairs.add((previous, character))
        previous = character
    return pairs


def basic_clean(text) -> str:
    """Normalize text before byte-pair encoding."""
    return ftfy.fix_text(html.unescape(html.unescape(text))).strip()


def whitespace_clean(text) -> str:
    """Collapse repeated whitespace before byte-pair encoding."""
    return re.sub(r"\s+", " ", text).strip()


class Tokenizer:
    """Byte-pair tokenizer compatible with the original CLIP model."""

    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {value: key for key, value in self.byte_encoder.items()}
        with gzip.open(bpe_path) as bpe_file:
            merges = bpe_file.read().decode("utf-8").split("\n")
        merges = merges[1:49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab += [value + "</w>" for value in vocab]
        vocab += ["".join(merge) for merge in merges]
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {value: key for key, value in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            index = 0
            while index < len(word):
                try:
                    next_index = word.index(first, index)
                    new_word.extend(word[index:next_index])
                    index = next_index
                except ValueError:
                    new_word.extend(word[index:])
                    break
                if (
                    word[index] == first
                    and index < len(word) - 1
                    and word[index + 1] == second
                ):
                    new_word.append(first + second)
                    index += 2
                else:
                    new_word.append(word[index])
                    index += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        result = " ".join(word)
        self.cache[token] = result
        return result

    def encode(self, text: str) -> List[int]:
        tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            tokens.extend(self.encoder[item] for item in self.bpe(token).split(" "))
        return tokens

    def decode(self, tokens: List[int]) -> str:
        text = "".join(self.decoder[token] for token in tokens)
        return (
            bytearray(self.byte_decoder[character] for character in text)
            .decode("utf-8", errors="replace")
            .replace("</w>", " ")
        )

    def encode_text(
        self,
        texts: Union[str, Iterable[str]],
        context_length: int = 77,
        truncate: bool = False,
    ) -> np.ndarray:
        """Tokenize text into fixed-length CLIP input arrays."""
        if isinstance(texts, str):
            texts = [texts]
        sot_token = self.encoder["<|startoftext|>"]
        eot_token = self.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self.encode(text) + [eot_token] for text in texts]
        result = np.zeros((len(all_tokens), context_length), dtype=np.int32)
        for index, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if not truncate:
                    raise RuntimeError(
                        f"Input {texts[index]} is too long for context length {context_length}"
                    )
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            result[index, :len(tokens)] = np.array(tokens)
        return result


__all__ = ["Clip", "Preprocessor", "Tokenizer"]

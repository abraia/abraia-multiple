import numpy as np
from abraia.inference import ops, Clip
from abraia.inference.detect import preprocess
from abraia.inference.ocr import TextRecognizer
from abraia.inference.sam import SAM
from abraia.inference.service import InferenceService


def test_softmax_values():
    logits = np.array([0, 10, -10])
    assert np.isclose(np.sum(ops.softmax(logits)), 1)


def test_clip_import():
    assert Clip is not None


def test_mask_to_polygon_accepts_boolean_masks():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 6:14] = True

    polygon = ops.mask_to_polygon(mask)

    assert len(polygon) >= 3


from abraia.runtime.stream import load_images_opencv
import cv2

def test_load_images_opencv(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test.jpg"
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.imwrite(str(p), img)

    images = load_images_opencv(str(p))
    assert len(images) == 1
    assert images[0].shape == (10, 10, 3)

    images_dir = load_images_opencv(str(d))
    assert len(images_dir) == 1
    assert images_dir[0].shape == (10, 10, 3)


def test_inference_service_reuses_and_closes_backend_sessions():
    instances = []

    class FakeModel:
        def __init__(self, uri):
            self.uri = uri
            self.closed = False
            instances.append(self)

        def run(self, image):
            return self.uri, image.shape

        def close(self):
            self.closed = True

    service = InferenceService({"fake": FakeModel})
    image = np.zeros((4, 5, 3), dtype=np.uint8)

    assert service.run("model-a", image, backend="fake") == ("model-a", (4, 5, 3))
    assert service.run("model-a", image, backend="fake") == ("model-a", (4, 5, 3))
    assert len(instances) == 1

    service.close()
    assert instances[0].closed


def test_classifier_preprocess_returns_fixed_input_shape_for_wide_images():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)

    assert preprocess(image, (224, 224)).shape == (1, 3, 224, 224)


def test_text_recognizer_processes_all_recognition_batches():
    class Input:
        name = "input"

    class Session:
        def get_inputs(self):
            return [Input()]

        def run(self, _outputs, inputs):
            batch = next(iter(inputs.values())).shape[0]
            return [np.zeros((batch, 1, 2), dtype=np.float32)]

    recognizer = TextRecognizer.__new__(TextRecognizer)
    recognizer.rec_image_shape = [3, 32, 320]
    recognizer.rec_batch_num = 6
    recognizer.limited_max_width = 1280
    recognizer.limited_min_width = 16
    recognizer.session = Session()
    recognizer.resize_norm_img = lambda image, ratio: np.zeros(
        (3, 32, 320), dtype=np.float32
    )
    recognizer.postprocess_op = lambda outputs: [
        ("decoded", 1.0)
    ] * len(outputs)

    images = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(7)]

    result = recognizer(images)

    assert len(result) == 7
    assert all(text == "decoded" for text, _score in result)


def test_sam_close_releases_both_sessions():
    class Session:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    sam = SAM.__new__(SAM)
    sam.encoder = Session()
    sam.decoder = Session()
    sam.image_embedding = object()
    sam._closed = False

    sam.close()

    assert sam.encoder is None
    assert sam.decoder is None
    assert sam.image_embedding is None
    assert sam._closed

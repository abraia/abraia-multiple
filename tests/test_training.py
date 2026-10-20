import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from abraia.dataset.annotation import Annotator, annotate_image
from abraia.dataset import create_dataset, delete_dataset
from abraia.dataset.remote import Dataset

from abraia.training import Training
from abraia.training import artifacts
from abraia.training.model_data import (
    metric_average,
    normalize_custom_models,
    normalize_model_record,
)
from abraia.dataset.preparation import prepare_dataset
from abraia.dataset.splitting import split_annotation_records
from abraia.training.classify import train_model


def test_annotator_uses_the_transformers_grounding_dino_model(monkeypatch):
    calls = []

    class FakePipeline:
        def __call__(self, image, candidate_labels, threshold):
            calls.append((image, candidate_labels, threshold))
            return [{
                "label": "cat",
                "score": 0.8,
                "box": {"xmin": 1, "ymin": 2, "xmax": 4, "ymax": 6},
            }]

        def close(self):
            calls.append("closed")

    class FakeTransformers:
        @staticmethod
        def pipeline(task, model):
            assert task == "zero-shot-object-detection"
            assert model == "IDEA-Research/grounding-dino-tiny"
            return FakePipeline()

    monkeypatch.setitem(sys.modules, "transformers", FakeTransformers)

    annotator = Annotator()
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    assert annotator.detect(image, [" Cat ", "dog."], threshold=0.4) == [{
        "label": "cat",
        "score": 0.8,
        "box": [1, 2, 3, 4],
    }]
    annotator.close()

    assert calls[0][0].size == (8, 8)
    assert calls[0][0].mode == "RGB"
    assert calls[0][1] == ["cat.", "dog."]
    assert calls[0][2] == 0.4
    assert calls[-1] == "closed"


def test_annotator_classifies_from_the_highest_scoring_detected_box(monkeypatch):
    annotator = object.__new__(Annotator)
    annotator.detect = lambda image, classes, threshold=0.3: [
        {"label": "dog", "score": 0.61, "box": [1, 2, 3, 4]},
        {"label": "cat", "score": 0.92, "box": [5, 6, 7, 8]},
    ]

    assert annotator.classify(np.zeros((8, 8, 3), dtype=np.uint8), ["cat", "dog"]) == [{
        "label": "cat",
        "score": 0.92,
    }]


def test_annotate_image_classification_keeps_only_the_detected_label():
    class FakeAnnotator:
        def classify(self, image, classes):
            return [{"label": "cat", "score": 0.9}]

        def detect(self, image, classes):
            raise AssertionError("classification should not call detect directly")

    result = annotate_image(
        {"url": "image.jpg", "name": "image.jpg"},
        ["cat"],
        annotator=FakeAnnotator(),
        classification=True,
        image_loader=lambda value: np.zeros((8, 8, 3), dtype=np.uint8),
        url_loader=lambda value: value,
    )

    assert result["objects"] == [{"label": "cat", "score": 0.9}]


@patch('abraia.dataset.remote.Dataset._load_annotations')
@patch('abraia.dataset.remote.Dataset._list_images')
@patch('abraia.dataset.remote.list_datasets')
def test_dataset_load(mock_list_datasets, mock_list_images, mock_load_annotations):
    mock_list_datasets.return_value = ['test_project']
    mock_load_annotations.return_value = [{'filename': 'test.jpg', 'objects': [{'label': 'cat'}]}]
    mock_list_images.return_value = [{'name': 'test.jpg'}]
    
    ds = Dataset('test_project', client=object())
    ds.load()
    
    assert ds.project == 'test_project'
    assert ds.annotations == [{'filename': 'test.jpg', 'objects': [{ 'label': 'cat' }]}]
    assert ds.classes == ['cat']
    assert ds.task == 'classification'
    assert ds.images == [{'name': 'test.jpg'}]
    mock_list_datasets.assert_called_once_with(ds.client)
    mock_load_annotations.assert_called_once_with()
    mock_list_images.assert_called_once_with()


def test_model_metric_helpers_normalize_catalog_payloads():
    assert metric_average([0.8, 1.0]) == 0.9
    assert metric_average([]) is None
    assert normalize_model_record("model.onnx") == {
        "name": "model.onnx",
        "metrics": {},
    }
    assert normalize_model_record({
        "model": "model.onnx",
        "metrics": {"acc": [0.8, 1.0]},
        "classes": ["cat"],
        "task": "classify",
    }) == {
        "name": "model.onnx",
        "metrics": {"acc": [0.8, 1.0]},
        "classes": ["cat"],
        "task": "classification",
    }
def test_dataset_save():
    class FakeClient:
        def __init__(self):
            self.saved = None

        def save_json(self, path, value):
            self.saved = (path, value)

    client = FakeClient()
    ds = Dataset('test_project', client=client)
    ds.annotations = [{'filename': 'test.jpg', 'objects': []}]

    ds.save()

    assert client.saved == ('test_project/annotations.json', ds.annotations)


@patch('abraia.dataset.remote.Dataset.save')
@patch('abraia.dataset.remote.load_url')
@patch('abraia.dataset.remote.load_image')
@patch('abraia.dataset.remote.Annotator')
def test_dataset_annotate_filter(
    mock_annotator_cls, mock_load_image, mock_load_url, mock_save
):
    ds = Dataset('test_project', client=object())
    ds.images = [
        {'name': 'old.jpg', 'url': 'old-url'},
        {'name': 'new.jpg', 'url': 'new-url'},
    ]
    ds.annotations = [{'filename': 'old.jpg', 'objects': [{'label': 'dog'}]}]

    mock_annotator_cls.return_value.detect.return_value = [
        {'label': 'cat', 'box': [0, 0, 1, 1]}
    ]
    mock_load_url.side_effect = lambda url: url
    mock_load_image.return_value = object()
    events = []

    annotations = ds.annotate('cat', callback=events.append)

    assert len(annotations) == 2
    assert annotations[0] == {'filename': 'old.jpg', 'objects': [{'label': 'dog'}]}
    assert annotations[1]['filename'] == 'new.jpg'
    assert annotations[1]['objects'][0]['label'] == 'cat'
    assert events == [{'current': 1, 'total': 1, 'filename': 'new.jpg'}]
    mock_annotator_cls.assert_called_once_with(segment=False)
    mock_save.assert_called_once_with()


def test_segmentation_annotation_converts_float_boxes_to_polygons():
    annotator = object.__new__(Annotator)

    class FakeSam:
        def encode(self, _image):
            return None

        def predict(self, image, prompt):
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[2:8, 3:10] = 255
            return mask

    annotator.sam = FakeSam()
    objects = [{"label": "car", "box": [3.2, 2.4, 6.5, 5.8]}]
    image = np.zeros((12, 16, 3), dtype=np.uint8)

    result = annotator.segment(image, objects)

    assert len(result[0]["polygon"]) >= 3
    assert min(point[0] for point in result[0]["polygon"]) >= 3
    assert min(point[1] for point in result[0]["polygon"]) >= 2


def test_dataset_annotated_status():
    ds = Dataset('test_project', client=object())
    # Empty dataset
    assert ds.annotated is False

    # Images present, all annotated
    ds.images = [{'name': 'img1.jpg'}, {'name': 'img2.jpg'}]
    ds.annotations = [{'filename': 'img1.jpg'}, {'filename': 'img2.jpg'}]
    ds._refresh_annotation_metadata()
    assert ds.annotated is True

    # Images present, not all annotated
    ds.annotations = [{'filename': 'img1.jpg'}]
    ds._refresh_annotation_metadata()
    assert ds.annotated is False


def test_dataset_prunes_orphaned_annotations():
    dataset = Dataset("demo", client=object())
    dataset.images = [{"name": "present.jpg"}]
    dataset.annotations = [
        {"filename": "present.jpg", "objects": []},
        {"filename": "deleted.jpg", "objects": []},
    ]

    assert dataset.prune_orphaned_annotations() == 1
    assert dataset.annotations == [{"filename": "present.jpg", "objects": []}]


def test_prepare_dataset_reports_download_progress(tmp_path, monkeypatch):
    import abraia.dataset.preparation as preparation

    annotations = [
        {"filename": "one.jpg"},
        {"filename": "two.jpg"},
        {"filename": "three.jpg"},
    ]
    client = object()
    dataset = type(
        "Dataset",
        (),
        {
            "project": str(tmp_path / "progress-test"),
            "annotations": annotations,
            "classes": ["cat"],
            "task": "classification",
            "client": client,
            "split": SimpleNamespace(
                options={},
                assignments=lambda: (annotations[:1], annotations[1:2], annotations[2:]),
            ),
            "prune_orphaned_annotations": lambda self: 0,
        },
    )()
    monkeypatch.setattr(preparation, "_resolve_client", lambda _client: client)
    monkeypatch.setattr(
        preparation, "save_data", lambda *args, **kwargs: None
    )
    events = []

    prepare_dataset(dataset, callback=events.append)

    assert [event["current"] for event in events][0] == 0
    assert sorted(event["current"] for event in events[1:]) == [1, 2, 3]
    assert all(event["total"] == 3 for event in events)


def test_prepare_dataset_reuses_splits_until_annotations_change(tmp_path, monkeypatch):
    import abraia.dataset.preparation as preparation

    project = tmp_path / "cached-training"
    annotation = {"filename": "one.jpg", "objects": []}
    dataset = SimpleNamespace(
        project=str(project),
        annotations=[annotation],
        classes=["cat"],
        task="classification",
        client=None,
        split=SimpleNamespace(
            options={},
            assignments=lambda: (dataset.annotations, [], []),
        ),
        prune_orphaned_annotations=lambda: 0,
    )
    downloads = []

    def save_data(_annotation, folder, *_args, **_kwargs):
        downloads.append(folder)
        os.makedirs(folder, exist_ok=True)

    monkeypatch.setattr(preparation, "save_data", save_data)
    monkeypatch.setattr(preparation, "_resolve_client", lambda _client: object())

    prepare_dataset(dataset)
    first_download_count = len(downloads)
    prepare_dataset(dataset)
    assert len(downloads) == first_download_count

    dataset.annotations = [{"filename": "two.jpg", "objects": []}]
    prepare_dataset(dataset)
    assert len(downloads) == first_download_count + 1


def test_dataset_preserves_first_seen_class_order():
    classes, task = Dataset._process_annotations(
        [{"objects": [{"label": "dog"}, {"label": "cat"}, {"label": "dog"}]}]
    )

    assert classes == ["dog", "cat"]
    assert task == "classification"


def test_dataset_uses_injected_client_without_global_client_calls():
    class FakeClient:
        userid = "user"

        def load_json(self, path):
            assert path == "project/annotations.json"
            return [{"filename": "cat.jpg", "objects": [{"label": "cat"}]}]

        def list_files(self, path):
            assert path == "project/"
            return [
                [{
                    "name": "cat.jpg",
                    "path": "project/cat.jpg",
                    "type": "image/jpeg",
                }],
                [],
            ]

    dataset = Dataset("project", client=FakeClient()).load(validate=False)

    assert dataset.classes == ["cat"]
    assert dataset.task == "classification"
    assert dataset.images[0]["url"]


def test_split_annotation_records_are_balanced_deterministic_and_local_rng():
    annotations = [
        {"filename": f"cat-{index}.jpg", "objects": [{"label": "cat"}]}
        for index in range(10)
    ] + [
        {"filename": f"dog-{index}.jpg", "objects": [{"label": "dog"}]}
        for index in range(10)
    ] + [{"filename": "background.jpg", "objects": []}]

    options = {
        "train_ratio": 0.6,
        "validation_ratio": 0.2,
        "test_ratio": 0.2,
        "random_state": 9,
    }
    np.random.seed(7)
    expected_random_value = np.random.random()
    np.random.seed(7)

    train, validation, test = split_annotation_records(annotations, **options)

    for split in (train, validation, test):
        assert len({item["filename"] for item in split}) == len(split)
    for label, expected in (
        ("cat", (6, 2, 2)),
        ("dog", (6, 2, 2)),
    ):
        counts = tuple(
            sum(
                any(obj.get("label") == label for obj in item.get("objects", []))
                for item in split
            )
            for split in (train, validation, test)
        )
        assert counts == expected

    assert split_annotation_records(annotations, **options) == (train, validation, test)
    assert any(item["filename"] == "background.jpg" for item in train)
    assert np.random.random() == expected_random_value


@patch("abraia.dataset.remote.annotate_image")
@patch("abraia.dataset.remote.Annotator")
@patch("abraia.dataset.remote.Dataset.save")
def test_dataset_annotate_uses_explicit_classification_task(
    mock_save, mock_annotator_cls, mock_annotate_image
):
    dataset = Dataset("demo", client=object())
    dataset.task = ""
    dataset.images = [{"name": "cat.jpg", "url": "cat-url"}]
    mock_annotate_image.return_value = {
        "filename": "cat.jpg",
        "objects": [{"label": "cat"}],
    }
    progress = []

    dataset.annotate(
        "cat",
        task="classification",
        include_empty=False,
        progress_callback=lambda *event: progress.append(event),
        is_cancelled=lambda: False,
    )

    assert mock_annotate_image.call_args.kwargs["classification"] is True
    assert mock_annotate_image.call_args.kwargs["segment"] is False
    mock_annotator_cls.return_value.close.assert_called_once_with()
    assert progress == [(1, 1, "cat.jpg", 1)]
    assert dataset.annotations == [mock_annotate_image.return_value]
    mock_save.assert_called_once_with()


def test_training_uses_public_training_workflow_defaults(monkeypatch):
    from abraia.training import service

    client = object()
    dataset = type(
        "Dataset",
        (),
        {"task": "classification", "classes": ["cat"], "client": client},
    )()
    operation_calls = []

    def train_operation(*args, **options):
        operation_calls.append((args, options))
        return {"stats": {"acc": 0.95}, "saved_model": {"name": "model_v1.onnx"}}

    monkeypatch.setattr(service, "load_dataset", lambda project, client: dataset)
    training = Training(client, train_operation=train_operation)
    training.list_models = lambda project: []

    result = training.train("test_proj", model_size="medium")

    args, options = operation_calls[0]
    assert args[:3] == ("test_proj", dataset, service.DEFAULT_TRAINING_EPOCHS["classification"])
    assert options["model_size"] == "medium"
    assert options["imgsz"] == service.DEFAULT_TRAINING_IMAGE_SIZE["classification"]
    assert result["stats"]["acc"] == 0.95
    assert result["model_name"] == "model_v1.onnx"


def test_ultralytics_model_compiles_native_hailo_bundle(tmp_path):
    from abraia.training.detect import Model

    bundle = tmp_path / "yolov8n_hailo_model"
    bundle.mkdir()
    (bundle / "yolov8n.hef").write_bytes(b"hef")
    (bundle / "metadata.yaml").write_text("task: detect\n")

    class FakeUltralyticsModel:
        def __init__(self):
            self.options = None

        def export(self, **options):
            self.options = options
            return str(bundle)

    class FakeClient:
        def __init__(self):
            self.uploads = []
            self.manifests = []

        def upload_file(self, source, path):
            self.uploads.append((source, path))

        def save_json(self, path, value):
            self.manifests.append((path, value))

    model = object.__new__(Model)
    model.model = FakeUltralyticsModel()
    model.model_name = "yolov8n"
    model.task = "detection"
    model.imgsz = 640
    model.client = FakeClient()
    model.metrics = {"mAP": 0.8}
    model.model_version = 2

    result = model.compile(
        "project",
        ["cat"],
        device="hailo8l",
        conf=0.3,
        iou=0.6,
    )

    assert model.model.options == {
        "format": "hailo",
        "name": "hailo8l",
        "quantize": 8,
        "imgsz": 640,
        "data": "project/data.yaml",
        "conf": 0.3,
        "iou": 0.6,
    }
    assert result["hef"] == "project/yolov8n_v2_hailo8l.hef"
    assert result["target"] == "hailo8l"
    assert model.hailo_model == result


def test_hailo_compile_stops_after_stage_when_cancelled(tmp_path, monkeypatch):
    from types import ModuleType
    from abraia.training.detect import Model

    calls = []
    cancelled = [False]

    class FakeClientRunner:
        def translate_onnx_model(self):
            calls.append("translate")

        def optimize(self):
            calls.append("optimize")
            cancelled[0] = True

        def compile(self):
            calls.append("compile")

    fake_sdk = ModuleType("hailo_sdk_client")
    fake_sdk.ClientRunner = FakeClientRunner
    monkeypatch.setitem(sys.modules, "hailo_sdk_client", fake_sdk)

    class FakeUltralyticsModel:
        def export(self, **_options):
            runner = FakeClientRunner()
            runner.translate_onnx_model()
            runner.optimize()
            runner.compile()

    model = object.__new__(Model)
    model.model = FakeUltralyticsModel()
    model.model_name = "yolov8n"
    model.task = "detection"
    model.imgsz = 640
    model.client = object()
    model.metrics = {}
    progress = []

    with pytest.raises(RuntimeError, match="Hailo compilation canceled"):
        model.compile(
            "project",
            ["cat"],
            progress_callback=lambda *event: progress.append(event),
            is_cancelled=lambda: cancelled[0],
        )

    assert calls == ["translate", "optimize"]
    assert not any(event[2] == "Optimizing INT8 model complete" for event in progress)


@patch("abraia.dataset.preparation.prepare_dataset")
@patch("abraia.training.classify.Model")
def test_training_service_reports_stage_transitions_before_backend_work(
    mock_model_cls, _mock_prepare_dataset, monkeypatch
):
    model = mock_model_cls.return_value
    model.test.return_value = {"acc": 1.0}
    model.save.return_value = {
        "onnx": "project/yolov8n_v1.onnx",
        "metadata": "project/yolov8n_v1.json",
        "version": 1,
    }
    dataset = type(
        "Dataset",
        (),
        {"task": "classification", "classes": ["cat"], "client": object()},
    )()
    from abraia.training import service

    monkeypatch.setattr(service, "load_dataset", lambda _project, client: dataset)
    events = []
    is_cancelled = lambda: False
    training = Training(dataset.client)
    training.list_models = lambda _project: []

    result = training.train(
        "project",
        epochs=2,
        training_callback=events.append,
        is_cancelled=is_cancelled,
        analyze_thresholds=False,
    )

    assert [event["stage"] for event in events] == [
        "Preparing dataset",
        "Loading model",
        "Training",
        "Validating model",
        "Exporting model",
    ]
    model.test.assert_called_once_with(
        split="test",
        is_cancelled=is_cancelled,
    )
    model.save.assert_called_once_with("project", ["cat"], device="cpu")
    assert result["saved_model"]["onnx"] == "project/yolov8n_v1.onnx"


def test_train_model_honors_cancellation_between_batches():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(2, 2)

        def forward(self, x):
            return self.fc(x)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.randn(4, 2), torch.tensor([0, 1, 0, 1])
        ),
        batch_size=2,
    )
    try:
        train_model(
            DummyModel(),
            {"train": loader, "val": loader},
            num_epochs=3,
            is_cancelled=lambda: True,
        )
    except RuntimeError as error:
        assert str(error) == "Training canceled"
    else:
        raise AssertionError("training did not honor cancellation")


def test_train_model_epochs():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(2, 2)
        def forward(self, x):
            return self.fc(x)

    model = DummyModel()
    inputs = torch.randn(4, 2)
    labels = torch.tensor([0, 1, 0, 1])
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    dataloaders = {'train': loader, 'val': loader}

    callback_calls = []
    def callback(progress):
        callback_calls.append(progress)

    num_epochs = 3
    train_model(model, dataloaders, num_epochs=num_epochs, callback=callback)

    assert len(callback_calls) == num_epochs
    for i, call in enumerate(callback_calls):
        assert call['epoch'] == i
        assert call['epochs'] == num_epochs


def test_classification_dataset_includes_held_out_test_split(tmp_path):
    from abraia.training.classify import Model

    image_folder = SimpleNamespace(classes=["cat", "dog"])
    model = object.__new__(Model)
    model.imgsz = 224
    model.transform = object()

    with patch(
        "abraia.training.classify.datasets.ImageFolder",
        return_value=image_folder,
    ) as image_folder_cls, patch(
        "abraia.training.classify.torch.utils.data.DataLoader",
        side_effect=lambda dataset, **_kwargs: dataset,
    ):
        dataloaders, classes = model.create_dataset(str(tmp_path), batch=4)

    assert set(dataloaders) == {"train", "val", "test"}
    assert classes == ["cat", "dog"]
    assert [call.args[0] for call in image_folder_cls.call_args_list] == [
        str(tmp_path / "train"),
        str(tmp_path / "val"),
        str(tmp_path / "test"),
    ]



def test_normalize_custom_models_adds_project_scope_and_model_defaults():
    models = normalize_custom_models(
        [
            {"name": " detector.onnx ", "task": "detection"},
            {"name": "classifier.onnx", "task": "classification"},
            {"name": "   "},
        ],
        project="demo",
        userid="user",
    )

    assert models == [
        {
            "name": "detector.onnx",
            "uri": "user/demo/detector.onnx",
            "task": "detection",
            "kind": "yolov8",
        },
        {
            "name": "classifier.onnx",
            "uri": "user/demo/classifier.onnx",
            "task": "classification",
            "kind": "resnet",
        },
    ]


def test_remote_dataset_saves_its_current_annotations():
    saved = []
    dataset = Dataset(
        "demo",
        SimpleNamespace(save_json=lambda path, value: saved.append((path, value))),
    )
    dataset.annotations = [{"filename": "photo.jpg", "objects": []}]

    dataset.save()

    assert saved == [("demo/annotations.json", dataset.annotations)]


@patch("abraia.dataset.remote.load_dataset")
def test_create_dataset_uses_the_normal_loader_and_saves_annotations(mock_load):
    calls = []
    dataset = Dataset("demo", SimpleNamespace())
    dataset.save = lambda: calls.append("saved")
    mock_load.return_value = dataset
    client = SimpleNamespace()
    assert create_dataset("demo", client=client) is dataset
    mock_load.assert_called_once_with("demo", client=client)
    assert calls == ["saved"]


def test_delete_dataset_removes_files_from_nested_project_folders():
    files_by_folder = {
        "demo/": (
            [{"path": "demo/annotations.json"}],
            [{"path": "demo/models", "name": "models"}],
        ),
        "demo/models/": ([{"path": "demo/models/model.onnx"}], []),
    }
    removed = []
    progress = []
    client = SimpleNamespace(
        list_files=lambda path: files_by_folder[path],
        remove_file=lambda path: removed.append(path),
    )

    assert delete_dataset(
        "demo", client=client,
        progress_callback=lambda *event: progress.append(event),
    ) == "demo"
    assert removed == [
        "demo/annotations.json", "demo/models/model.onnx"
    ]
    assert progress[0] == (0, 2, "Preparing deletion", False)
    assert progress[-1] == (2, 2, "demo/models/model.onnx", True)

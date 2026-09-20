from abraia.training.dataset import Dataset
from abraia.training.dataset import Annotator
from abraia.datasets import DatasetBase
from abraia.training.orchestration import split_dataset
from unittest.mock import patch
import numpy as np


@patch('abraia.training.dataset.Dataset._load_annotations')
@patch('abraia.training.dataset.Dataset._list_images')
@patch('abraia.training.dataset.list_datasets')
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
    mock_load_annotations.assert_called_once_with('test_project')
    mock_list_images.assert_called_once_with('test_project')


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


@patch('abraia.training.dataset.Dataset.save')
@patch('abraia.training.dataset.load_url')
@patch('abraia.training.dataset.load_image')
@patch('abraia.training.dataset.Annotator')
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
    ds._update_annotated()
    assert ds.annotated is True

    # Images present, not all annotated
    ds.annotations = [{'filename': 'img1.jpg'}]
    ds._update_annotated()
    assert ds.annotated is False


def test_prune_orphaned_annotations_removes_missing_images_and_saves():
    from abraia.training import prune_orphaned_annotations

    saved = []
    dataset = type(
        "Dataset",
        (),
        {
            "images": [{"name": "present.jpg"}],
            "annotations": [
                {"filename": "present.jpg", "objects": []},
                {"filename": "deleted.jpg", "objects": []},
            ],
            "save": lambda self: saved.append(list(self.annotations)),
        },
    )()

    assert prune_orphaned_annotations(dataset) == 1
    assert dataset.annotations == [{"filename": "present.jpg", "objects": []}]
    assert saved == [dataset.annotations]


def test_prepare_dataset_reports_download_progress(monkeypatch):
    import abraia.training as training
    import abraia.training.orchestration as orchestration

    annotations = [
        {"filename": "one.jpg"},
        {"filename": "two.jpg"},
        {"filename": "three.jpg"},
    ]
    client = type(
        "Client",
        (),
        {"check_file": lambda self, _path: False},
    )()
    dataset = type(
        "Dataset",
        (),
        {
            "project": "progress-test",
            "images": [
                {"name": "one.jpg"},
                {"name": "two.jpg"},
                {"name": "three.jpg"},
            ],
            "annotations": annotations,
            "classes": ["cat"],
            "task": "classification",
            "client": client,
        },
    )()
    monkeypatch.setattr(orchestration.os.path, "exists", lambda _path: False)
    monkeypatch.setattr(
        orchestration,
        "split_dataset",
        lambda values: (values[:1], values[1:2], values[2:]),
    )
    monkeypatch.setattr(orchestration, "save_data", lambda *args: None)
    events = []

    training.prepare_dataset(dataset, callback=events.append)

    assert [event["current"] for event in events][0] == 0
    assert sorted(event["current"] for event in events[1:]) == [1, 2, 3]
    assert all(event["total"] == 3 for event in events)


def test_dataset_base_preserves_first_seen_class_order():
    classes, task = DatasetBase._process_annotations(
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


def test_split_dataset_does_not_change_numpy_global_rng_state():
    values = [{"objects": [{"label": "cat"}]} for _ in range(8)]
    np.random.seed(7)
    expected = np.random.random()
    np.random.seed(7)

    split_dataset(values)

    assert np.random.random() == expected


def test_split_dataset_balances_each_class_using_requested_ratios():
    annotations = [
        {"filename": f"cat-{index}.jpg", "objects": [{"label": "cat"}]}
        for index in range(10)
    ] + [
        {"filename": f"dog-{index}.jpg", "objects": [{"label": "dog"}]}
        for index in range(10)
    ]

    train, validation, test = split_dataset(
        annotations,
        {
            "train_ratio": 0.6,
            "validation_ratio": 0.2,
            "test_ratio": 0.2,
            "random_state": 9,
        },
    )

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


def test_split_dataset_is_reproducible_and_keeps_background_in_train():
    annotations = [
        {"filename": f"cat-{index}.jpg", "objects": [{"label": "cat"}]}
        for index in range(8)
    ] + [{"filename": "background.jpg", "objects": []}]

    first = split_dataset(annotations)
    second = split_dataset(annotations)

    assert first == second
    assert any(item["filename"] == "background.jpg" for item in first[0])


from abraia.training import ModelTrainer
from abraia.training.service import TrainingService

@patch('abraia.training.classify.Model')
def test_model_trainer_test(mock_classify_model_cls):
    mock_model = mock_classify_model_cls.return_value
    mock_model.test.return_value = {'acc': 0.95, 'confusionMatrix': [[10, 0], [1, 9]]}
    
    trainer = ModelTrainer('test_proj', 'classification', ['cat', 'dog'])
    metrics = trainer.test('val')
    
    assert metrics['acc'] == 0.95
    mock_model.test.assert_called_once_with(split='val')


@patch('abraia.training.detect.Model')
def test_model_trainer_passes_large_size_to_detection_training(mock_detect_model_cls):
    ModelTrainer(
        'test_proj',
        'segmentation',
        ['cat'],
        model_size='large',
    )

    mock_detect_model_cls.assert_called_once_with(
        'segment',
        imgsz=640,
        client=None,
        model_size='large',
    )


@patch('abraia.training.classify.Model')
def test_model_trainer_passes_medium_size_to_classification_training(mock_classify_model_cls):
    ModelTrainer(
        'test_proj',
        'classification',
        ['cat'],
        model_size='medium',
    )

    mock_classify_model_cls.assert_called_once_with(
        client=None,
        model_size='medium',
    )


def test_ultralytics_model_compiles_native_hailo_bundle(tmp_path):
    from abraia.training.detect import Model

    calibration_data = tmp_path / "data.yaml"
    calibration_data.write_text("train: images\n")
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
        calibration_data=calibration_data,
        fraction=0.5,
        conf=0.3,
        iou=0.6,
    )

    assert model.model.options == {
        "format": "hailo",
        "name": "hailo8l",
        "quantize": 8,
        "imgsz": 640,
        "data": str(calibration_data),
        "fraction": 0.5,
        "conf": 0.3,
        "iou": 0.6,
    }
    assert result["hef"] == "project/yolov8n_v2_hailo8l.hef"
    assert result["target"] == "hailo8l"
    assert model.hailo_model == result


@patch('abraia.training.prepare_dataset')
@patch('abraia.training.ModelTrainer')
def test_training_service_reports_stage_transitions_before_backend_work(
    mock_trainer_cls, mock_prepare_dataset
):
    trainer = mock_trainer_cls.return_value
    trainer.test.return_value = {"acc": 1.0}
    trainer.save.return_value = {
        "onnx": "project/yolov8n_v1.onnx",
        "metadata": "project/yolov8n_v1.json",
        "version": 1,
    }
    dataset = type(
        "Dataset",
        (),
        {"task": "classification", "classes": ["cat"], "client": object()},
    )()
    events = []
    is_cancelled = lambda: False

    result = TrainingService().train_dataset(
        "project",
        dataset,
        epochs=2,
        training_callback=events.append,
        is_cancelled=is_cancelled,
    )

    assert [event["stage"] for event in events] == [
        "Preparing dataset",
        "Loading model",
        "Training",
        "Validating model",
        "Exporting model",
    ]
    trainer.test.assert_called_once_with(
        split="test",
        is_cancelled=is_cancelled,
    )
    mock_trainer_cls.assert_called_once_with(
        "project", "classification", ["cat"], client=dataset.client
    )
    assert result["saved_model"]["onnx"] == "project/yolov8n_v1.onnx"


def test_train_model_honors_cancellation_between_batches():
    import torch

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


import torch
from abraia.training.classify import train_model

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
    from types import SimpleNamespace
    from unittest.mock import patch

    from abraia.training.classify import Model

    image_folder = SimpleNamespace(classes=["cat", "dog"])
    model = object.__new__(Model)
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

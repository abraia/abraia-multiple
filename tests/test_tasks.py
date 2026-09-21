from abraia.tasks import (
    HAILO_TASKS,
    MODEL_SIZES,
    PIPELINE_TASKS,
    TRAINING_TASKS,
    normalize_model_size,
    normalize_task,
    to_hailo_task,
    to_ultralytics_task,
)


def test_task_vocabulary_is_canonical_across_public_apis():
    assert PIPELINE_TASKS == (
        "classification",
        "detection",
        "segmentation",
        "pose",
        "recognition",
    )
    assert set(TRAINING_TASKS) <= set(PIPELINE_TASKS)
    assert set(HAILO_TASKS) <= set(PIPELINE_TASKS)


def test_task_names_normalize_to_the_canonical_spelling():
    assert normalize_task(" detection ") == "detection"
    assert normalize_task("detect") == "detection"


def test_task_aliases_are_normalized_at_api_boundaries():
    assert [normalize_task(value) for value in (
        "classify", "detect", "segment", "recognize"
    )] == [
        "classification", "detection", "segmentation", "recognition"
    ]


def test_task_aliases_are_supported_by_backend_adapters():
    assert to_hailo_task("segment") == "segment"
    assert to_ultralytics_task("classify") == "classify"


def test_canonical_task_values_translate_at_backend_boundary():
    assert to_hailo_task("segmentation") == "segment"
    assert to_ultralytics_task("detection") == "detect"


def test_training_model_sizes_are_shared_and_normalized():
    assert MODEL_SIZES == ("small", "medium", "large")
    assert normalize_model_size(" Medium ") == "medium"

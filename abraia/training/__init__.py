"""Public training APIs and compatibility exports."""

from .annotations import (
    annotation_counts,
    canonical_filename,
    dataset_has_annotations,
    find_image,
    image_filename,
    objects_for_image,
    upsert_annotation,
)
from .core import RemoteDataset
from .dataset import download_file, list_datasets, list_models, load_dataset, search_images
from .orchestration import (
    ModelTrainer,
    prepare_dataset,
    save_annotation,
    save_config,
    save_data,
    split_dataset,
)
from .service import TrainingService


__all__ = [
    "ModelTrainer",
    "RemoteDataset",
    "TrainingService",
    "annotation_counts",
    "canonical_filename",
    "dataset_has_annotations",
    "download_file",
    "find_image",
    "image_filename",
    "list_datasets",
    "list_models",
    "load_dataset",
    "objects_for_image",
    "prepare_dataset",
    "save_annotation",
    "save_config",
    "save_data",
    "search_images",
    "split_dataset",
    "upsert_annotation",
]

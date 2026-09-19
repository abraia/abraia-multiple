"""Public training APIs and compatibility exports."""

from .annotations import (
    annotation_counts,
    canonical_filename,
    dataset_has_annotations,
    find_image,
    image_filename,
    objects_for_image,
    prune_orphaned_annotations,
    upsert_annotation,
)
from .core import save_versioned_model, versioned_model_paths
from .dataset import (
    download_file,
    list_datasets,
    list_model_records,
    list_models,
    load_dataset,
    search_images,
)
from .orchestration import (
    ModelTrainer,
    dataset_split_summary,
    prepare_dataset,
    save_annotation,
    save_config,
    save_data,
    split_dataset,
)
from .splitting import (
    DATASET_SPLITS,
    DEFAULT_EVALUATION_SPLIT,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SPLIT_RATIOS,
    DEFAULT_SPLIT_OPTIONS,
    SPLIT_NAMES,
    SPLIT_RATIO_KEYS,
    normalize_split_ratios,
    split_annotation_records,
    summarize_split_records,
)
from .service import DatasetProjectService, TrainingService


__all__ = [
    "ModelTrainer",
    "DEFAULT_RANDOM_STATE",
    "DATASET_SPLITS",
    "DEFAULT_EVALUATION_SPLIT",
    "DEFAULT_SPLIT_RATIOS",
    "DEFAULT_SPLIT_OPTIONS",
    "TrainingService",
    "DatasetProjectService",
    "SPLIT_NAMES",
    "SPLIT_RATIO_KEYS",
    "annotation_counts",
    "canonical_filename",
    "dataset_has_annotations",
    "download_file",
    "dataset_split_summary",
    "find_image",
    "image_filename",
    "list_datasets",
    "list_models",
    "list_model_records",
    "load_dataset",
    "objects_for_image",
    "prune_orphaned_annotations",
    "prepare_dataset",
    "save_annotation",
    "save_config",
    "save_data",
    "search_images",
    "save_versioned_model",
    "split_dataset",
    "split_annotation_records",
    "summarize_split_records",
    "normalize_split_ratios",
    "versioned_model_paths",
    "upsert_annotation",
]

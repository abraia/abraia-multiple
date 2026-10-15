"""Public training APIs."""

from .artifacts import (
    HAILO_EXPORT_TARGETS,
    list_model_records,
    list_models,
    save_hailo_bundle,
    save_versioned_model,
    versioned_model_paths,
)
from .thresholds import detection_threshold_analysis
from .orchestration import (
    ModelTrainer,
    dataset_split_summary,
    download_file,
    prepare_dataset,
    save_annotation,
    save_config,
    save_data,
    split_dataset,
    train_dataset,
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
from .model_data import (
    metric_average,
    normalize_confusion_matrix,
    normalize_custom_models,
    normalize_model_record,
)


__all__ = [
    "ModelTrainer",
    "DEFAULT_RANDOM_STATE",
    "HAILO_EXPORT_TARGETS",
    "DATASET_SPLITS",
    "DEFAULT_EVALUATION_SPLIT",
    "DEFAULT_SPLIT_RATIOS",
    "DEFAULT_SPLIT_OPTIONS",
    "SPLIT_NAMES",
    "SPLIT_RATIO_KEYS",
    "download_file",
    "dataset_split_summary",
    "detection_threshold_analysis",
    "list_models",
    "list_model_records",
    "metric_average",
    "normalize_model_record",
    "normalize_custom_models",
    "normalize_confusion_matrix",
    "prepare_dataset",
    "save_annotation",
    "save_config",
    "save_data",
    "save_hailo_bundle",
    "save_versioned_model",
    "split_dataset",
    "train_dataset",
    "split_annotation_records",
    "summarize_split_records",
    "normalize_split_ratios",
    "versioned_model_paths",
]

"""MMDetection adapter for RTMDet detection and instance segmentation."""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from ..inference.models.detection import prepare_rtmdet_input
from .artifacts import (
    HAILO_EXPORT_TARGETS,
    _resolve_client,
    save_hailo_bundle,
    save_versioned_model,
)
from .model_data import metric_average
from ..dataset.splitting import DEFAULT_EVALUATION_SPLIT


MODEL_CONFIGS = {
    "detection": {
        "small": "rtmdet_tiny_8xb32-300e_coco",
        "medium": "rtmdet_s_8xb32-300e_coco",
        "large": "rtmdet_l_8xb32-300e_coco",
    },
    "segmentation": {
        "small": "rtmdet-ins_tiny_8xb32-300e_coco",
        "medium": "rtmdet-ins_s_8xb32-300e_coco",
        "large": "rtmdet-ins_l_8xb32-300e_coco",
    },
}


def _mmdet_modules():
    try:
        from mmengine.config import Config
        from mmengine.hooks import Hook
        from mmengine.runner import Runner
        from mmdet.apis import inference_detector
    except ImportError as error:
        raise RuntimeError(
            "RTMDet requires MMDetection and MMCV. Follow the RTMDet "
            "installation instructions in the Abraia README."
        ) from error
    return Config, Hook, Runner, inference_detector


def _download_config(config_name, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    config_path = next(
        (path for path in destination.rglob("*.py") if path.stem == config_name),
        None,
    )
    weight_path = next(destination.rglob("*.pth"), None)
    if config_path is None or weight_path is None:
        command = [
            shutil.which("mim") or "mim", "download", "mmdet",
            "--config", config_name, "--dest", str(destination),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(
                f"Unable to download the RTMDet config/checkpoint {config_name}. "
                f"Check that OpenMIM is installed and network access is available. {detail}"
            )
        if config_path is None:
            config_path = next(
                (path for path in destination.rglob("*.py") if path.stem == config_name),
                None,
            )
        weight_path = next(destination.rglob("*.pth"), None)
    try:
        import mmdet

        package_dir = Path(mmdet.__file__).parent
        config_candidates = (
            package_dir / ".mim" / "configs" / "rtmdet" / f"{config_name}.py",
            package_dir / "configs" / "rtmdet" / f"{config_name}.py",
            package_dir.parent / "configs" / "rtmdet" / f"{config_name}.py",
        )
        installed_config = next(
            (path for path in config_candidates if path.is_file()), None
        )
        if installed_config:
            config_path = installed_config
    except ImportError:
        pass
    if config_path is None:
        raise RuntimeError(
            f"OpenMIM did not provide a config for {config_name}."
        )
    return config_path, weight_path


def _deploy_config_path(task):
    """Find the MMDeploy ONNX Runtime config for an RTMDet task."""
    if task == "detection":
        relative_path = Path("mmdet/detection/detection_onnxruntime_static.py")
    else:
        relative_path = Path(
            "mmdet/instance-seg/"
            "instance-seg_rtmdet-ins_onnxruntime_static-640x640.py"
        )
    roots = []
    if os.environ.get("MMDEPLOY_DIR"):
        roots.append(Path(os.environ["MMDEPLOY_DIR"]))
    try:
        import mmdeploy

        package_dir = Path(mmdeploy.__file__).parent
        roots.extend((package_dir, package_dir / ".mim", package_dir.parent))
    except ImportError as error:
        raise RuntimeError(
            "RTMDet Hailo compilation requires MMDeploy. Install a compatible "
            "MMDeploy build and set MMDEPLOY_DIR to its source directory if "
            "its deployment configs are not installed with the package."
        ) from error

    candidates = [root / "configs" / relative_path for root in roots]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        "Could not find the MMDeploy ONNX Runtime config needed for Hailo. "
        "Install MMDeploy with its configs or set MMDEPLOY_DIR to the "
        "MMDeploy source directory."
    )


def _export_raw_onnx(model, task, imgsz, destination):
    """Export RTMDet's raw heads with PyTorch's ONNX exporter."""
    import torch

    class RawRTMDet(torch.nn.Module):
        def __init__(self, detector, segmentation):
            super().__init__()
            self.detector = detector
            self.segmentation = segmentation

        def forward(self, image):
            features = self.detector.extract_feat(image)
            outputs = self.detector.bbox_head(features)
            class_scores, box_distances = outputs[:2]
            values = list(class_scores) + list(box_distances)
            if self.segmentation:
                mask_kernels, mask_features = outputs[2:]
                values.extend(mask_kernels)
                values.append(mask_features)
            return tuple(values)

    model.eval()
    wrapper = RawRTMDet(model, task == "segmentation").eval()
    strides = [int(value[0]) for value in model.bbox_head.prior_generator.strides]
    num_levels = len(strides)
    output_names = (
        [f"cls_{index}" for index in range(num_levels)]
        + [f"bbox_{index}" for index in range(num_levels)]
    )
    if task == "segmentation":
        output_names += [f"kernel_{index}" for index in range(num_levels)]
        output_names.append("mask_features")

    sample = torch.zeros((1, 3, imgsz, imgsz), dtype=torch.float32)
    with torch.no_grad():
        outputs = wrapper(sample)
    torch.onnx.export(
        wrapper,
        sample,
        str(destination),
        input_names=["input"],
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    layout = {
        "classScores": output_names[:num_levels],
        "boxDistances": output_names[num_levels:2 * num_levels],
        "strides": strides,
        "bboxDistancesScaled": True,
        "scoreThreshold": 0.001,
        "nmsPre": 30000,
        "nmsThreshold": 0.65,
        "maxDetections": 300,
    }
    if task == "segmentation":
        kernel_start = 2 * num_levels
        layout.update({
            "maskKernels": output_names[kernel_start:kernel_start + num_levels],
            "maskFeatures": output_names[-1],
            "dyconvChannels": int(model.bbox_head.dyconv_channels),
            "dyconvLayers": int(model.bbox_head.num_dyconvs),
            "maskFeatureStride": int(imgsz // outputs[-1].shape[-1]),
        })
    return output_names, layout


def _xywh_from_polygon(points):
    xs = points[0::2]
    ys = points[1::2]
    if len(xs) < 3 or len(ys) < 3:
        return None
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return [x1, y1, x2 - x1, y2 - y1]


def _write_coco_annotations(project, split, classes, task):
    """Convert Abraia's prepared YOLO split into MMDetection's COCO format."""
    split_dir = Path(project) / split
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    images = []
    annotations = []
    image_id = 1
    annotation_id = 1

    for image_path in sorted(path for path in images_dir.iterdir() if path.is_file()):
        with Image.open(image_path) as image:
            width, height = image.size
        images.append({
            "id": image_id,
            "file_name": image_path.name,
            "width": width,
            "height": height,
        })
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.is_file():
            with label_path.open(encoding="utf-8") as label_file:
                for line in label_file:
                    fields = line.split()
                    if len(fields) < 5:
                        continue
                    class_id = int(float(fields[0]))
                    if not 0 <= class_id < len(classes):
                        continue
                    values = [float(value) for value in fields[1:]]
                    polygon = None
                    if task == "segmentation" and len(values) >= 6 and len(values) % 2 == 0:
                        polygon = [
                            coordinate
                            for point_index, value in enumerate(values)
                            for coordinate in [
                                value * (width if point_index % 2 == 0 else height)
                            ]
                        ]
                        bbox = _xywh_from_polygon(polygon)
                        if bbox is None:
                            continue
                        segmentation = [polygon]
                    else:
                        center_x, center_y, box_width, box_height = values[:4]
                        box_width *= width
                        box_height *= height
                        x = center_x * width - box_width / 2
                        y = center_y * height - box_height / 2
                        bbox = [x, y, box_width, box_height]
                        segmentation = []
                    annotations.append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id + 1,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                        "segmentation": segmentation,
                    })
                    annotation_id += 1
        image_id += 1

    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": index + 1, "name": name, "supercategory": "object"}
            for index, name in enumerate(classes)
        ],
    }
    annotation_path = split_dir / "annotations.coco.json"
    with annotation_path.open("w", encoding="utf-8") as annotation_file:
        json.dump(payload, annotation_file)
    return annotation_path


def _set_dataset(config, split, annotation_path, classes):
    config.data_root = os.path.abspath(os.path.dirname(os.path.dirname(annotation_path)))
    dataset = getattr(config, f"{split}_dataloader").dataset
    dataset.type = "CocoDataset"
    dataset.ann_file = f"{split}/annotations.coco.json"
    dataset.data_prefix = dict(img=f"{split}/images/")
    dataset.metainfo = dict(classes=tuple(classes))
    dataset.data_root = config.data_root


def _set_pipeline_size(pipeline, imgsz):
    for transform in pipeline or []:
        transform_type = transform.get("type")
        if transform_type == "RandomResize":
            transform.scale = (imgsz * 2, imgsz * 2)
        elif transform_type in ("CachedMosaic", "CachedMixUp"):
            transform.img_scale = (imgsz, imgsz)
        elif transform_type == "RandomCrop":
            transform.crop_size = (imgsz, imgsz)
        elif transform_type == "Pad":
            transform.size = (imgsz, imgsz)
        elif transform_type == "Resize":
            transform.scale = (imgsz, imgsz)


def _metric_value(metrics, *keys):
    for key in keys:
        value = metric_average(metrics.get(key)) if isinstance(metrics, dict) else None
        if value is not None:
            return value
    return 0.0


def _latest_scalar(runner, tag):
    try:
        value = runner.message_hub.get_scalar(tag).current()
        return float(value)
    except (KeyError, AttributeError, TypeError, ValueError):
        return 0.0


class Model:
    """Adapt MMDetection's RTMDet models to Abraia's training interface."""

    def __init__(
        self, task, imgsz=640, client=None, model_size="small", checkpoint=None,
        classes=None,
    ):
        self.task = task
        self.model_size = model_size
        self.imgsz = int(imgsz)
        self.client = _resolve_client(client)
        self.checkpoint = checkpoint
        self.classes = list(classes or [])
        self.metrics = {}
        self.model = None
        self.runner = None
        self.model_version = None
        self.model_name = "rtmdet" + ("_ins" if self.task == "segmentation" else "")
        self.model_name += f"_{self.model_size}"

    def _prepare_config(self, project, classes, batch, epochs):
        Config, _, _, _ = _mmdet_modules()
        config_name = MODEL_CONFIGS[self.task][self.model_size]
        model_dir = Path(project) / ".rtmdet" / self.task / self.model_size
        config_path, pretrained_checkpoint = _download_config(config_name, model_dir)
        config = Config.fromfile(str(config_path))

        annotation_paths = {
            split: _write_coco_annotations(project, split, classes, self.task)
            for split in ("train", "val", "test")
        }
        for split, annotation_path in annotation_paths.items():
            _set_dataset(config, split, annotation_path, classes)
            _set_pipeline_size(
                getattr(config, f"{split}_dataloader").dataset.pipeline,
                self.imgsz,
            )
        if "train_pipeline_stage2" in config:
            _set_pipeline_size(config.train_pipeline_stage2, self.imgsz)

        config.model.bbox_head.num_classes = len(classes)
        config.train_dataloader.batch_size = batch
        config.train_dataloader.num_workers = min(
            int(config.train_dataloader.get("num_workers", 2)), 4
        )
        config.train_cfg.max_epochs = epochs
        config.train_cfg.val_interval = 1
        config.default_hooks.checkpoint.interval = max(1, epochs)
        config.default_hooks.checkpoint.max_keep_ckpts = 1
        config.work_dir = str(model_dir / "work_dirs")
        config.load_from = None
        self.config = config
        self.work_dir = Path(config.work_dir)
        self.classes = list(classes)
        self.pretrained_checkpoint = self.checkpoint or pretrained_checkpoint

    def train(self, project, epochs=100, batch=8, callback=None, is_cancelled=None):
        _, Hook, Runner, _ = _mmdet_modules()
        epochs = int(epochs or 100)
        self._prepare_config(project, self.classes, batch, epochs)

        is_cancelled = is_cancelled or (lambda: False)
        callback = callback or (lambda progress: None)

        class ProgressHook(Hook):
            def before_train_iter(_hook, runner, batch_idx, data_batch=None):
                if is_cancelled():
                    raise RuntimeError("Training canceled")

            def after_train_epoch(_hook, runner):
                if is_cancelled():
                    raise RuntimeError("Training canceled")
                callback({
                    "epoch": runner.epoch,
                    "epochs": epochs,
                    "loss": _latest_scalar(runner, "train/loss"),
                    "acc": _latest_scalar(runner, "coco/bbox_mAP"),
                })

        self.runner = Runner.from_cfg(self.config)
        if self.pretrained_checkpoint:
            import torch

            try:
                checkpoint = torch.load(
                    self.pretrained_checkpoint,
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                checkpoint = torch.load(
                    self.pretrained_checkpoint, map_location="cpu"
                )
            state_dict = checkpoint.get("state_dict", checkpoint)
            model_state = self.runner.model.state_dict()
            compatible_state = {}
            for name, value in state_dict.items():
                key = name[7:] if name.startswith("module.") else name
                if key in model_state and model_state[key].shape == value.shape:
                    compatible_state[key] = value
            self.runner.model.load_state_dict(compatible_state, strict=False)
        self.runner.register_hook(ProgressHook(), priority="LOW")
        self.runner.train()
        self.model = self.runner.model

    def test(self, split=DEFAULT_EVALUATION_SPLIT, is_cancelled=None):
        if self.runner is None:
            raise RuntimeError("Train the RTMDet model before evaluating it.")
        if is_cancelled and is_cancelled():
            raise RuntimeError("Training canceled")
        if split == "test":
            metrics = self.runner.test()
        elif split in ("val", "validation", "valid"):
            metrics = self.runner.val()
        else:
            raise ValueError("RTMDet evaluation supports the val and test splits.")
        box_map = _metric_value(metrics, "coco/bbox_mAP")
        box_map50 = _metric_value(metrics, "coco/bbox_mAP_50", box_map)
        mask_map = _metric_value(metrics, "coco/segm_mAP")
        mask_map50 = _metric_value(metrics, "coco/segm_mAP_50", mask_map)
        self.metrics = {
            "mAP": mask_map50 if self.task == "segmentation" else box_map50,
            "box_mAP": box_map50,
        }
        if self.task == "segmentation":
            self.metrics["mask_mAP"] = mask_map50
        return self.metrics

    def save(self, project, classes, device="cpu"):
        if self.runner is None:
            raise RuntimeError("Train the RTMDet model before saving it.")
        work_dir = Path(self.runner.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_name = "final.pth"
        checkpoint_path = work_dir / checkpoint_name
        config_path = work_dir / "final.py"
        self.config.dump(str(config_path))
        self.runner.save_checkpoint(
            str(work_dir),
            filename=checkpoint_name,
            save_optimizer=False,
            save_param_scheduler=False,
        )
        return self.export_onnx(
            project,
            classes,
            checkpoint=checkpoint_path,
            config_path=config_path,
            model=self.runner.model,
            upload_project=project,
        )

    def export_onnx(
        self,
        project,
        classes,
        checkpoint=None,
        config_path=None,
        model=None,
        upload_project=None,
    ):
        """Export a trained or pretrained RTMDet checkpoint with PyTorch."""
        try:
            import onnx
            from mmengine.config import Config
            from mmdet.utils import register_all_modules
            from mmdet.registry import MODELS
        except ImportError as error:
            raise RuntimeError(
                "RTMDet ONNX export requires PyTorch, MMDetection, MMCV, "
                "MMEngine, and ONNX."
            ) from error

        project_dir = Path(project)
        model_dir = project_dir / ".rtmdet" / self.task / self.model_size
        if config_path is None or checkpoint is None:
            config_path, pretrained_checkpoint = _download_config(
                MODEL_CONFIGS[self.task][self.model_size], model_dir
            )
            checkpoint = checkpoint or self.checkpoint or pretrained_checkpoint
        config_path = Path(config_path)
        checkpoint = Path(checkpoint)
        if not config_path.is_file():
            raise FileNotFoundError(f"RTMDet config does not exist: {config_path}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"RTMDet checkpoint does not exist: {checkpoint}")

        config = Config.fromfile(str(config_path))
        register_all_modules(init_default_scope=True)
        if model is None:
            model = MODELS.build(config.model)
            import torch

            try:
                checkpoint_data = torch.load(
                    str(checkpoint), map_location="cpu", weights_only=False
                )
            except TypeError:
                checkpoint_data = torch.load(str(checkpoint), map_location="cpu")
            model.load_state_dict(
                checkpoint_data.get("state_dict", checkpoint_data), strict=False
            )
        model.to("cpu").eval()
        work_dir = Path(model_dir) / "work_dirs"
        work_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = work_dir / "end2end.onnx"
        output_names, raw_layout = _export_raw_onnx(
            model, self.task, self.imgsz, onnx_path
        )
        onnx.checker.check_model(onnx.load(str(onnx_path)))
        result = save_versioned_model(
            self.client,
            upload_project or project,
            self.model_name,
            str(onnx_path),
            {
                "format": "onnx",
                "backend": "rtmdet",
                "task": self.task,
                "inputShape": [1, 3, self.imgsz, self.imgsz],
                "classes": classes,
                "metrics": self.metrics,
                "rawOutputs": raw_layout,
                "outputNames": output_names,
                "modelSize": self.model_size,
            },
            artifacts={
                "checkpoint": checkpoint,
                "config": config_path,
            },
        )
        self.model_version = result["version"]
        self.onnx_path = onnx_path
        return result

    def compile(
        self, project, classes, device="hailo8",
        conf=None, iou=None, progress_callback=None, is_cancelled=None,
        config_path=None,
    ):
        """Compile a trained RTMDet checkpoint through the Hailo DFC."""
        if conf is not None and not 0 <= float(conf) <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1.")
        if iou is not None and not 0 <= float(iou) <= 1:
            raise ValueError("IoU threshold must be between 0 and 1.")
        target = str(device or "hailo8").strip().lower()
        if target not in HAILO_EXPORT_TARGETS:
            choices = ", ".join(HAILO_EXPORT_TARGETS)
            raise ValueError(f"Unsupported Hailo target '{target}'. Use: {choices}")

        project_dir = Path(project)
        model_dir = project_dir / ".rtmdet" / self.task / self.model_size
        model_work_dir = model_dir / "work_dirs"
        checkpoint_path = Path(self.checkpoint or model_work_dir / "final.pth")
        config_path = Path(config_path or model_work_dir / "final.py")
        if not checkpoint_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(
                "RTMDet Hailo compilation requires the trained final.pth "
                "and final.py. Save the trained model first."
            )

        image_dirs = [project_dir / split / "images" for split in ("train", "val")]
        image_paths = []
        for image_dir in image_dirs:
            if image_dir.is_dir():
                image_paths = sorted(
                    path for path in image_dir.rglob("*")
                    if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
                )
                if image_paths:
                    break
        if not image_paths:
            raise ValueError(
                "RTMDet Hailo compilation needs calibration images in the "
                "project's train or validation split."
            )

        if progress_callback:
            progress_callback(0, 100, "Checking Hailo compiler", "Running")
        if is_cancelled and is_cancelled():
            raise RuntimeError("Hailo compilation canceled")
        try:
            import cv2
            import onnx
            from hailo_sdk_client import ClientRunner
        except ImportError as error:
            raise RuntimeError(
                "RTMDet Hailo compilation requires ONNX, OpenCV, and the "
                "matching Hailo Dataflow Compiler."
            ) from error

        try:
            from mmdeploy.apis import torch2onnx
        except ImportError as error:
            raise RuntimeError(
                "RTMDet Hailo compilation requires MMDeploy to create a "
                "temporary ONNX graph from the trained checkpoint."
            ) from error
        sample_image = next(
            (
                path
                for split in ("train", "val", "test")
                for path in sorted((project_dir / split / "images").glob("*"))
                if path.is_file()
            ),
            None,
        )
        if sample_image is None:
            raise RuntimeError("RTMDet ONNX export requires a dataset image.")
        temporary_export = tempfile.TemporaryDirectory(
            prefix="abraia-rtmdet-hailo-"
        )
        temporary_dir = Path(temporary_export.name)
        source = temporary_dir / "compile.onnx"
        torch2onnx(
            str(sample_image),
            str(temporary_dir),
            source.name,
            str(_deploy_config_path(self.task)),
            str(config_path),
            str(checkpoint_path),
            device="cpu",
        )
        if not source.is_file():
            raise RuntimeError(
                "MMDeploy did not produce the temporary ONNX graph needed "
                "for Hailo compilation."
            )

        graph = onnx.load(str(source)).graph
        graph_inputs = [
            item for item in graph.input
            if item.name not in {initializer.name for initializer in graph.initializer}
        ]
        if len(graph_inputs) != 1:
            raise RuntimeError(
                "Hailo RTMDet compilation requires an ONNX graph with one "
                "image input; found {}.".format(len(graph_inputs))
            )
        input_info = graph_inputs[0]
        dimensions = [dimension.dim_value for dimension in input_info.type.tensor_type.shape.dim]
        if len(dimensions) != 4 or any(value <= 0 for value in dimensions):
            raise RuntimeError(
                "Hailo RTMDet compilation requires a static NCHW ONNX input."
            )
        batch, channels, height, width = dimensions
        if batch != 1 or channels != 3:
            raise RuntimeError(
                "Hailo RTMDet compilation requires an ONNX input shaped "
                "[1, 3, height, width]."
            )
        work_dir = model_dir / "hailo" / target
        work_dir.mkdir(parents=True, exist_ok=True)
        selected_paths = image_paths[:128]
        calibration = []
        for image_path in selected_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            calibration.append(
                prepare_rtmdet_input(image, (1, 3, height, width))[0]
            )
        if not calibration:
            raise ValueError("No readable images were found for Hailo calibration.")
        calibration_data_array = np.asarray(calibration, dtype=np.float32)

        if self.model_version is None:
            version_suffix = source.stem.rsplit("_v", 1)
            if len(version_suffix) == 2 and version_suffix[1].isdigit():
                self.model_version = int(version_suffix[1])
        model_name = f"{self.model_name}_v{self.model_version or 'custom'}"
        # Compile tensors before ONNX NMS so the RTMDet host decoder can apply
        # confidence filtering and class-aware NMS after Hailo inference.
        end_nodes = [output.name for output in graph.output]
        output_roles = {}
        center_point_box = 0
        for role, aliases in {
            "dets": ("dets", "detections"),
            "labels": ("labels", "classes"),
            "masks": ("masks", "mask"),
        }.items():
            output = next(
                (
                    item.name for item in graph.output
                    if any(alias in item.name.lower() for alias in aliases)
                ),
                None,
            )
            if output:
                output_roles[role] = output
        if self.task == "detection":
            nms_node = next(
                (node for node in graph.node if node.op_type == "NonMaxSuppression"),
                None,
            )
            if nms_node is not None:
                for attribute in nms_node.attribute:
                    if attribute.name == "center_point_box":
                        center_point_box = int(attribute.i)
                produced_values = {
                    output_name
                    for node in graph.node
                    for output_name in node.output
                }
                nms_inputs = [
                    name for name in nms_node.input
                    if name in produced_values
                ]
                if len(nms_inputs) >= 2:
                    end_nodes = nms_inputs[:2]
                    output_roles = {
                        "boxes": end_nodes[0],
                        "scores": end_nodes[1],
                    }
        runner = ClientRunner(hw_arch=target)
        if progress_callback:
            progress_callback(10, 100, "Parsing RTMDet ONNX graph", "Running")
        try:
            runner.translate_onnx_model(
                str(source),
                model_name,
                end_node_names=end_nodes,
            )
            runner.load_model_script(
                "\n".join((
                    "model_optimization_flavor(optimization_level=2)",
                    "post_quantization_optimization(finetune, policy=enabled, "
                    f"dataset_size={len(calibration_data_array)})",
                ))
            )
            if is_cancelled and is_cancelled():
                raise RuntimeError("Hailo compilation canceled")
            if progress_callback:
                progress_callback(35, 100, "Quantizing with calibration images", "Running")
            runner.optimize(calibration_data_array)
            if is_cancelled and is_cancelled():
                raise RuntimeError("Hailo compilation canceled")
            if progress_callback:
                progress_callback(75, 100, "Compiling HEF", "Running")
            hef = runner.compile()
        except Exception as error:
            raise RuntimeError(
                "Hailo could not compile the RTMDet ONNX graph. MMDeploy's "
                "ONNX includes detection post-processing, which may need a "
                "Hailo-specific graph cut or DFC layer support. Original "
                f"error: {error}"
            ) from error

        hef_path = work_dir / f"{self.model_name}_{target}.hef"
        hef_path.write_bytes(hef)
        bundle_dir = work_dir / f"{self.model_name}_{target}"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(hef_path, bundle_dir / hef_path.name)
        manifest = {
            "format": "hailo",
            "backend": "rtmdet",
            "target": target,
            "task": self.task,
            "inputShape": dimensions,
            "classes": classes,
            "metrics": self.metrics,
            "calibrationImages": len(calibration),
            "outputNodes": end_nodes,
            "outputRoles": output_roles,
            "centerPointBox": center_point_box,
            "nms": "host" if self.task == "detection" else False,
            "confidenceThreshold": 0.25 if conf is None else float(conf),
            "iouThreshold": 0.7 if iou is None else float(iou),
            "postprocessing": "rtmdet_host",
        }
        result = save_hailo_bundle(
            self.client,
            project,
            self.model_name,
            bundle_dir,
            target,
            version=self.model_version,
            metadata=manifest,
            progress_callback=progress_callback,
        )
        result["localBundle"] = str(bundle_dir)
        self.hailo_model = result
        return result

    def run(self, img):
        if self.model is None:
            raise RuntimeError("Train or load the RTMDet model before inference.")
        _, _, _, inference_detector = _mmdet_modules()
        result = inference_detector(self.model, img)
        prediction = result.pred_instances
        boxes = prediction.bboxes.detach().cpu().numpy()
        labels = prediction.labels.detach().cpu().numpy()
        scores = prediction.scores.detach().cpu().numpy()
        masks = getattr(prediction, "masks", None)
        objects = []
        for index, (box, class_id, score) in enumerate(zip(boxes, labels, scores)):
            x1, y1, x2, y2 = [int(round(value)) for value in box.tolist()]
            obj = {
                "label": self.classes[int(class_id)],
                "score": float(score),
                "box": [x1, y1, x2 - x1, y2 - y1],
            }
            if self.task == "segmentation" and masks is not None:
                mask = masks[index].detach().cpu().numpy().astype(np.uint8)
                import cv2

                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
                    obj["polygon"] = contour.tolist()
            objects.append(obj)
        return objects

"""LibreYOLO YOLOv9 detection training adapter."""

import os
from pathlib import Path

from ..dataset.splitting import DEFAULT_EVALUATION_SPLIT
from .artifacts import (
    _resolve_client,
    save_hailo_bundle,
    save_versioned_model,
)


YOLO9_HAILO_END_NODES = (
    "/head/cv2.0/cv2.0.2/Conv",
    "/head/cv3.0/cv3.0.2/Conv",
    "/head/cv2.1/cv2.1.2/Conv",
    "/head/cv3.1/cv3.1.2/Conv",
    "/head/cv2.2/cv2.2.2/Conv",
    "/head/cv3.2/cv3.2.2/Conv",
)


MODEL_SIZE_TYPES = {
    "small": "t",
    "medium": "s",
    "large": "c",
}
YOLO9_HAILO_TARGETS = ("hailo8", "hailo8l", "hailo10h")


def _metric(metrics, *names):
    for name in names:
        value = metrics.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


class Model:
    """Train and export a LibreYOLO YOLOv9 detection model."""

    def __init__(self, project, imgsz=640, client=None, model_size="small",
                 checkpoint=None):
        try:
            from libreyolo import LibreYOLO
        except ImportError as error:
            raise ImportError(
                "LibreYOLO training requires the optional dependency. "
                "Install abraia[libreyolo]."
            ) from error

        size = MODEL_SIZE_TYPES[model_size]
        self.model = LibreYOLO(checkpoint or f"LibreYOLO9{size}.pt")
        self.model_name = f"yolo9{size}"
        self.task = "detection"
        self.model_size = model_size
        self.imgsz = imgsz
        self.client = _resolve_client(client)
        self._project = Path(project).resolve()
        self.metrics = {}
        self.model_version = None
        self.checkpoint_path = None

    def train(self, project, epochs=100, batch=32, callback=None,
              is_cancelled=None):
        self._project = Path(project).resolve()
        output_dir = (
            self._project
            / ".abraia" / "training" / self.task / self.model_size
        )

        def on_epoch(event):
            if is_cancelled and is_cancelled():
                raise RuntimeError("Training canceled")
            val_metrics = getattr(event, "val_metrics", {}) or {}
            if callback:
                callback({
                    "epoch": int(event.epoch) + 1,
                    "epochs": int(event.total_epochs),
                    "loss": float(event.train_loss or 0.0),
                    "acc": _metric(val_metrics, "metrics/mAP50"),
                })

        result = self.model.train(
            data=str(Path(project) / "data.yaml"),
            epochs=epochs,
            batch=batch,
            imgsz=self.imgsz,
            project=str(output_dir),
            name="train",
            exist_ok=True,
            callbacks=on_epoch,
        )
        result = result if isinstance(result, dict) else {}
        checkpoint = (
            result.get("best_checkpoint")
            or result.get("last_checkpoint")
            or output_dir / "train" / "weights" / "best.pt"
        )
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            checkpoint = output_dir / "train" / "weights" / "last.pt"
        self.checkpoint_path = checkpoint.resolve()

    def test(self, split=DEFAULT_EVALUATION_SPLIT, is_cancelled=None):
        if is_cancelled and is_cancelled():
            raise RuntimeError("Training canceled")
        metrics = self.model.val(
            data=str(self._dataset_yaml()),
            split=split,
            imgsz=self.imgsz,
        )
        if is_cancelled and is_cancelled():
            raise RuntimeError("Training canceled")
        metrics = metrics if isinstance(metrics, dict) else {}
        self.metrics = {
            "mAP": _metric(metrics, "metrics/mAP50"),
            "P": _metric(metrics, "metrics/precision"),
            "R": _metric(metrics, "metrics/recall"),
        }
        return self.metrics

    def _dataset_yaml(self):
        # The model is trained and evaluated against the same prepared dataset.
        return self._project / "data.yaml"

    def save(
        self, project, classes, device="cpu", half=False,
        publish_common=False,
    ):
        self._project = Path(project).resolve()
        checkpoint = self.checkpoint_path or (
            self._project / ".abraia" / "training" / self.task
            / self.model_size / "train" / "weights" / "best.pt"
        )
        if not checkpoint.is_file():
            last_checkpoint = checkpoint.parent / "last.pt"
            checkpoint = last_checkpoint if last_checkpoint.is_file() else None
        if checkpoint is None:
            # ModelTrainer loads the default LibreYOLO checkpoint in __init__.
            # Saving without train() should export that loaded model directly.
            trained_model = self.model
            artifacts = None
        else:
            from libreyolo import LibreYOLO

            trained_model = LibreYOLO(str(checkpoint))
            artifacts = {"checkpoint": checkpoint}
        if not classes:
            model_classes = getattr(trained_model, "names", None)
            if isinstance(model_classes, dict):
                classes = [model_classes[index] for index in sorted(model_classes)]
            elif isinstance(model_classes, (list, tuple)):
                classes = list(model_classes)
            else:
                classes = []
        model_path = (
            self._project / ".abraia" / "training" / self.task
            / self.model_size / "train" / "weights"
            / f"{self.model_name}.onnx"
        )
        common_paths = None
        if publish_common:
            if getattr(self.client, "userid", None) != "multiple":
                raise ValueError(
                    "Publishing to the common model repository requires the "
                    "Abraia 'multiple' account from .env."
                )
            common_paths = {
                "onnx": f"models/{self.model_name}.onnx",
                "metadata": f"models/{self.model_name}.json",
            }
            common_uris = {
                key: f"multiple/{path}"
                for key, path in common_paths.items()
            }
            if any(
                self.client.check_file(path)
                for path in common_paths.values()
            ):
                raise FileExistsError(
                    "A common model asset already exists for "
                    f"{self.model_name}; refusing to overwrite it."
                )
        try:
            trained_model.export(
                format="onnx",
                imgsz=self.imgsz,
                nms=True,
                conf=0.001,
                iou=0.7,
                max_det=300,
                device=device,
                half=half,
                output_path=str(model_path),
            )
            result = save_versioned_model(
                self.client,
                project,
                self.model_name,
                str(model_path),
                {
                    "backend": "libreyolo",
                    "kind": "yolo9",
                    "task": self.task,
                    "inputShape": [1, 3, self.imgsz, self.imgsz],
                    "classes": classes,
                    "metrics": self.metrics,
                    "modelSize": self.model_size,
                },
                artifacts=artifacts,
            )
            self.model_version = result["version"]
            if common_paths:
                try:
                    self.client.upload_file(
                        str(model_path), common_paths["onnx"]
                    )
                    self.client.save_json(
                        common_paths["metadata"],
                        {
                            "backend": "libreyolo",
                            "kind": "yolo9",
                            "task": self.task,
                            "inputShape": [1, 3, self.imgsz, self.imgsz],
                            "classes": classes,
                            "metrics": self.metrics,
                            "modelSize": self.model_size,
                            "license": "MIT",
                        },
                    )
                except Exception:
                    try:
                        self.client.remove_file(common_paths["onnx"])
                    except Exception:
                        pass
                    raise
                result["common"] = common_uris
            return result
        finally:
            if model_path.is_file():
                try:
                    os.remove(model_path)
                except OSError:
                    pass

    def run(self, img):
        results = self.model(img)
        results = results[0] if isinstance(results, (list, tuple)) else results
        objects = []
        for box in results.boxes:
            class_id = int(box.cls)
            label = results.names[class_id]
            x1, y1, x2, y2 = box.xyxy.squeeze().tolist()
            objects.append({
                "label": label,
                "score": float(box.conf),
                "box": [
                    round(x1), round(y1),
                    round(x2) - round(x1), round(y2) - round(y1),
                ],
            })
        return objects

    def compile(
        self,
        project,
        classes,
        device="hailo8",
        conf=None,
        iou=None,
        progress_callback=None,
        is_cancelled=None,
    ):
        """Compile YOLOv9 to a Hailo HEF with raw detection heads."""
        target = str(device or "hailo8").strip().lower()
        if target not in YOLO9_HAILO_TARGETS:
            choices = ", ".join(YOLO9_HAILO_TARGETS)
            raise ValueError(f"Unsupported Hailo target '{target}'. Use: {choices}")
        if conf is not None and not 0 <= float(conf) <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1.")
        if iou is not None and not 0 <= float(iou) <= 1:
            raise ValueError("IoU threshold must be between 0 and 1.")

        if progress_callback:
            progress_callback(0, 100, "Checking Hailo compiler", "Running")
        if is_cancelled and is_cancelled():
            raise RuntimeError("Hailo compilation canceled")
        try:
            import numpy as np
            import onnx
            from PIL import Image
            from hailo_sdk_client import ClientRunner
            from libreyolo import LibreYOLO
        except ImportError as error:
            raise RuntimeError(
                "LibreYOLO Hailo compilation requires abraia[libreyolo], "
                "ONNX, Pillow, NumPy, and the matching Hailo Dataflow Compiler."
            ) from error

        checkpoint = self.checkpoint_path or (
            Path(project).resolve() / ".abraia" / "training" / self.task
            / self.model_size / "train" / "weights" / "best.pt"
        )
        if not checkpoint.is_file():
            checkpoint = checkpoint.parent / "last.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "LibreYOLO Hailo compilation requires the trained best.pt "
                "or last.pt checkpoint. Train and save the model first."
            )

        project_dir = Path(project).resolve()
        model_dir = (
            project_dir / ".abraia" / "training" / self.task / self.model_size
        )
        work_dir = model_dir / "hailo" / target
        work_dir.mkdir(parents=True, exist_ok=True)
        model_path = work_dir / f"{self.model_name}.onnx"
        compiled_model = LibreYOLO(str(checkpoint))
        if progress_callback:
            progress_callback(5, 100, "Exporting static ONNX graph", "Running")
        compiled_model.export(
            format="onnx",
            imgsz=self.imgsz,
            dynamic=False,
            simplify=True,
            output_path=str(model_path),
        )
        if is_cancelled and is_cancelled():
            raise RuntimeError("Hailo compilation canceled")

        graph = onnx.load(str(model_path)).graph
        input_dims = [
            dimension.dim_value
            for dimension in graph.input[0].type.tensor_type.shape.dim
        ]
        if len(input_dims) != 4 or any(value <= 0 for value in input_dims):
            raise RuntimeError(
                "LibreYOLO Hailo compilation requires a static NCHW ONNX input."
            )
        if input_dims[0] != 1 or input_dims[1] != 3:
            raise RuntimeError(
                "LibreYOLO Hailo compilation requires input shape [1, 3, H, W]."
            )
        _, _, height, width = input_dims
        image_paths = [
            path
            for split in ("train", "val")
            for path in sorted((project_dir / split / "images").rglob("*"))
            if path.is_file()
            and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        ]
        if not image_paths:
            raise ValueError(
                "LibreYOLO Hailo compilation needs calibration images in the "
                "project's train or validation split."
            )
        calibration = []
        for image_path in image_paths[:128]:
            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB").resize((width, height))
                    calibration.append(np.asarray(image, dtype=np.float32))
            except (OSError, ValueError):
                continue
        if not calibration:
            raise ValueError("No readable images were found for Hailo calibration.")
        calibration_data = np.asarray(calibration, dtype=np.float32)

        if progress_callback:
            progress_callback(15, 100, "Parsing YOLOv9 ONNX graph", "Running")
        runner = ClientRunner(hw_arch=target)
        model_name = f"{self.model_name}_v{self.model_version or 'custom'}"
        try:
            runner.translate_onnx_model(
                str(model_path),
                model_name,
                end_node_names=list(YOLO9_HAILO_END_NODES),
            )
            runner.load_model_script(
                "normalization1 = normalization([0.0, 0.0, 0.0], "
                "[255.0, 255.0, 255.0])"
            )
            if is_cancelled and is_cancelled():
                raise RuntimeError("Hailo compilation canceled")
            if progress_callback:
                progress_callback(
                    35, 100, "Quantizing with calibration images", "Running"
                )
            runner.optimize(calibration_data)
            if is_cancelled and is_cancelled():
                raise RuntimeError("Hailo compilation canceled")
            if progress_callback:
                progress_callback(75, 100, "Compiling HEF", "Running")
            hef_path = work_dir / f"{self.model_name}_{target}.hef"
            hef_path.write_bytes(runner.compile())
        except Exception as error:
            if isinstance(error, RuntimeError) and "canceled" in str(error).lower():
                raise
            raise RuntimeError(
                "Hailo could not compile the LibreYOLO YOLOv9 graph. Verify "
                "the DFC and Model Zoo versions, the YOLOv9 end-node names, "
                "and the calibration images. Original error: "
                f"{error}"
            ) from error

        if is_cancelled and is_cancelled():
            raise RuntimeError("Hailo compilation canceled")
        manifest = {
            "format": "hailo",
            "backend": "libreyolo",
            "kind": "yolo9",
            "target": target,
            "task": self.task,
            "inputShape": input_dims,
            "classes": classes,
            "metrics": self.metrics,
            "calibrationImages": len(calibration),
            "outputNodes": list(YOLO9_HAILO_END_NODES),
            "postprocessing": "raw_yolo9_heads",
            "confidenceThreshold": 0.25 if conf is None else float(conf),
            "iouThreshold": 0.7 if iou is None else float(iou),
        }
        result = save_hailo_bundle(
            self.client,
            project,
            self.model_name,
            work_dir,
            target,
            version=self.model_version,
            metadata=manifest,
            progress_callback=progress_callback,
        )
        result["localBundle"] = str(work_dir)
        self.hailo_model = result
        return result


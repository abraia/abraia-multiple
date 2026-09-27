"""Composable inference steps for multi-model runtime pipelines."""

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional


@dataclass
class RegionInput:
    """An image region that retains its source detection identity."""

    id: str
    image: Any
    source_box: tuple[float, float, float, float]
    parent_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)


def _as_box(value):
    """Return a numeric ``x, y, width, height`` box or raise clearly."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("Pipeline regions require a four-value 'box'")
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError("Pipeline region boxes must contain numbers") from error


def crop_region(image, box, padding=0.0):
    """Crop a clipped ``x, y, width, height`` region from an image."""
    source_box = _as_box(box)
    if not hasattr(image, "shape") or len(image.shape) < 2:
        raise ValueError("Pipeline crop inputs must expose an image shape")
    height, width = image.shape[:2]
    x, y, region_width, region_height = source_box
    padding = max(0.0, float(padding))
    x1 = max(0, int(round(x - region_width * padding)))
    y1 = max(0, int(round(y - region_height * padding)))
    x2 = min(width, int(round(x + region_width * (1 + padding))))
    y2 = min(height, int(round(y + region_height * (1 + padding))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Pipeline crop region is empty")
    return image[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)


def _resolve_reference(context, reference):
    """Resolve a step reference such as ``detect.results``."""
    reference = str(reference or "").strip()
    if reference == "frame":
        return context.frame
    if reference == "results":
        return context.results
    if reference in context.artifacts:
        return context.artifacts[reference]
    step_id, separator, output = reference.partition(".")
    if separator:
        key = f"{step_id}.{output}"
        if key in context.artifacts:
            return context.artifacts[key]
        if step_id in context.artifacts:
            return context.artifacts[step_id]
    raise ValueError(f"Unknown pipeline input reference: {reference}")


def _result_id(result, prefix, index):
    value = result.get("id") if isinstance(result, Mapping) else None
    return str(value or f"{prefix}-{index}")


def _normalize_results(value, prefix):
    if value is None:
        return []
    if not isinstance(value, list):
        return value
    normalized = []
    for index, result in enumerate(value):
        if isinstance(result, Mapping):
            result = dict(result)
            result.setdefault("id", _result_id(result, prefix, index))
        normalized.append(result)
    return normalized


def _region_result(value):
    """Normalize a model result while preserving OCR-like text results."""
    if isinstance(value, Mapping):
        result = dict(value)
    elif isinstance(value, list):
        result = {"items": value}
    else:
        result = {"value": value}
    if "text" not in result and isinstance(result.get("items"), list):
        text_items = [item for item in result["items"] if isinstance(item, Mapping)]
        texts = [str(item.get("text", "")) for item in text_items if item.get("text")]
        if texts:
            result["text"] = "\n".join(texts)
            scores = [float(item.get("score", 0.0)) for item in text_items if item.get("text")]
            if scores:
                result["text_score"] = max(scores)
    return result


def _map_region_result(value, source_box):
    """Translate region-local boxes and points back to frame coordinates."""
    if not isinstance(value, Mapping) or not source_box:
        return dict(value) if isinstance(value, Mapping) else value
    x, y = float(source_box[0]), float(source_box[1])
    mapped = dict(value)
    box = mapped.get("box")
    try:
        if len(box) == 4 and not isinstance(box[0], (list, tuple)):
            mapped["box"] = [
                x + float(box[0]),
                y + float(box[1]),
                float(box[2]),
                float(box[3]),
            ]
    except (TypeError, ValueError, IndexError):
        pass
    points = mapped.get("points")
    if hasattr(points, "copy") and getattr(points, "ndim", 0) == 2:
        points = points.astype(float, copy=True)
        points[:, 0] += x
        points[:, 1] += y
        mapped["points"] = points
    elif isinstance(points, list):
        mapped["points"] = [
            [x + float(point[0]), y + float(point[1])]
            for point in points
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
    if isinstance(mapped.get("items"), list):
        mapped["items"] = [
            _map_region_result(item, source_box)
            for item in mapped["items"]
        ]
    return mapped


class CompositionStep:
    """Base class for a named step in a composed pipeline."""

    def __init__(self, step_id):
        self.step_id = str(step_id)

    def close(self):
        return None


class ModelStep(CompositionStep):
    """Run a model on the frame or on each input region."""

    def __init__(self, step_id, model, input_ref="frame", model_kwargs=None):
        super().__init__(step_id)
        self.model = model
        self.input_ref = input_ref
        self.model_kwargs = dict(model_kwargs or {})

    def __call__(self, context):
        inputs = _resolve_reference(context, self.input_ref)
        if self.input_ref == "frame":
            output = self.model.run(inputs, **self.model_kwargs)
            output = _normalize_results(output, self.step_id)
            context.artifacts[f"{self.step_id}.results"] = output
            if isinstance(output, list):
                context.results = output
            return context

        if not isinstance(inputs, Iterable) or isinstance(inputs, (str, bytes)):
            raise ValueError(f"Model step '{self.step_id}' requires region inputs")
        regions = list(inputs)
        output = self._run_regions(regions)
        context.artifacts[f"{self.step_id}.results"] = output
        return context

    def _run_regions(self, regions):
        """Run this model over region inputs and preserve their identities."""
        if not regions:
            output = []
        elif callable(getattr(self.model, "run_batch", None)):
            raw_results = self.model.run_batch(
                [region.image for region in regions], **self.model_kwargs
            )
            if len(raw_results) != len(regions):
                raise ValueError(
                    f"Model step '{self.step_id}' returned {len(raw_results)} results "
                    f"for {len(regions)} regions"
                )
            output = [
                {
                    "item_id": region.id,
                    "parent_id": region.parent_id,
                    "source_box": list(region.source_box),
                    **_region_result(raw),
                }
                for region, raw in zip(regions, raw_results)
            ]
        else:
            output = []
            for region in regions:
                raw = self.model.run(region.image, **self.model_kwargs)
                output.append({
                    "item_id": region.id,
                    "parent_id": region.parent_id,
                    "source_box": list(region.source_box),
                    **_region_result(raw),
                })
        return output

    def close(self):
        close = getattr(self.model, "close", None)
        if callable(close):
            close()


class BoundModelStep(ModelStep):
    """Run a model on selected detections and attach results automatically."""

    def __init__(
        self,
        step_id,
        model,
        input_ref,
        labels=None,
        min_confidence=None,
        crop_padding=0.0,
        attach_to=None,
        field="result",
        model_kwargs=None,
    ):
        super().__init__(step_id, model, input_ref=input_ref, model_kwargs=model_kwargs)
        self.labels = {str(label) for label in labels or ()}
        self.min_confidence = min_confidence
        self.crop_padding = crop_padding
        self.attach_to = attach_to
        self.field = str(field)

    def __call__(self, context):
        values = _resolve_reference(context, self.input_ref)
        if not isinstance(values, list):
            raise ValueError(
                f"Bound model step '{self.step_id}' requires detection results"
            )
        selected = []
        for value in values:
            if not isinstance(value, Mapping):
                continue
            if self.labels and str(value.get("label", "")) not in self.labels:
                continue
            if self.min_confidence is not None:
                score = value.get("score", value.get("confidence", 0.0))
                if float(score) < float(self.min_confidence):
                    continue
            selected.append(value)

        regions = []
        spectral_cube = getattr(context, "spectral_cube", None)
        spectral_input = getattr(self.model, "input_kind", "image") == "spectral"
        if spectral_input and spectral_cube is None:
            raise ValueError(
                f"Bound model step '{self.step_id}' requires a spectral source"
            )
        for index, value in enumerate(selected):
            parent_id = str(value.get("id") or f"{self.step_id}-parent-{index}")
            crop_source = spectral_cube if spectral_input else context.frame
            cropped, source_box = crop_region(
                crop_source,
                value.get("box"),
                padding=self.crop_padding,
            )
            regions.append(RegionInput(
                id=f"{self.step_id}-{index}",
                image=cropped,
                source_box=source_box,
                parent_id=parent_id,
                metadata=dict(value),
            ))
        output = self._run_regions(regions)
        context.artifacts[f"{self.step_id}.results"] = output
        if self.attach_to is None:
            context.results = output
            return context

        targets = _resolve_reference(context, self.attach_to)
        if not isinstance(targets, list):
            raise ValueError(
                f"Bound model step '{self.step_id}' requires a result-list attach target"
            )
        by_id = {
            str(value.get("id")): value
            for value in targets
            if isinstance(value, Mapping) and value.get("id") is not None
        }
        for value in output:
            target = by_id.get(str(value.get("parent_id")))
            if target is None or not isinstance(target, dict):
                continue
            attached = _map_region_result(value, value.get("source_box"))
            attached.pop("item_id", None)
            attached.pop("parent_id", None)
            attached.pop("source_box", None)
            target[self.field] = attached
        context.artifacts[f"{self.step_id}.results"] = targets
        context.results = targets
        return context


class FilterStep(CompositionStep):
    """Filter detection-like records before a downstream region step."""

    def __init__(self, step_id, input_ref, labels=None, min_confidence=None):
        super().__init__(step_id)
        self.input_ref = input_ref
        self.labels = {str(label) for label in labels or ()}
        self.min_confidence = min_confidence

    def __call__(self, context):
        values = _resolve_reference(context, self.input_ref)
        if not isinstance(values, list):
            raise ValueError(f"Filter step '{self.step_id}' requires a result list")
        output = []
        for value in values:
            if not isinstance(value, Mapping):
                continue
            if self.labels and str(value.get("label", "")) not in self.labels:
                continue
            if self.min_confidence is not None:
                score = value.get("score", value.get("confidence", 0.0))
                if float(score) < float(self.min_confidence):
                    continue
            output.append(value)
        context.artifacts[f"{self.step_id}.results"] = output
        return context


class CropStep(CompositionStep):
    """Fan out detections into source-aware region inputs."""

    def __init__(self, step_id, input_ref, padding=0.0):
        super().__init__(step_id)
        self.input_ref = input_ref
        self.padding = padding

    def __call__(self, context):
        values = _resolve_reference(context, self.input_ref)
        if not isinstance(values, list):
            raise ValueError(f"Crop step '{self.step_id}' requires a result list")
        regions = []
        for index, value in enumerate(values):
            if not isinstance(value, Mapping):
                continue
            parent_id = str(value.get("id") or f"{self.step_id}-parent-{index}")
            cropped, source_box = crop_region(
                context.frame,
                value.get("box"),
                padding=self.padding,
            )
            regions.append(RegionInput(
                id=f"{self.step_id}-{index}",
                image=cropped,
                source_box=source_box,
                parent_id=parent_id,
                metadata=dict(value),
            ))
        context.artifacts[f"{self.step_id}.items"] = regions
        return context


class AttachStep(CompositionStep):
    """Attach per-region model output to its parent detection."""

    def __init__(self, step_id, input_ref, target_ref, field="result"):
        super().__init__(step_id)
        self.input_ref = input_ref
        self.target_ref = target_ref
        self.field = str(field)

    def __call__(self, context):
        values = _resolve_reference(context, self.input_ref)
        targets = _resolve_reference(context, self.target_ref)
        if not isinstance(values, list) or not isinstance(targets, list):
            raise ValueError(f"Attach step '{self.step_id}' requires result lists")
        by_id = {
            str(value.get("id")): value
            for value in targets
            if isinstance(value, Mapping) and value.get("id") is not None
        }
        for value in values:
            if not isinstance(value, Mapping):
                continue
            target = by_id.get(str(value.get("parent_id")))
            if target is None:
                continue
            if isinstance(target, dict):
                attached = _map_region_result(value, value.get("source_box"))
                attached.pop("item_id", None)
                attached.pop("parent_id", None)
                attached.pop("source_box", None)
                target[self.field] = attached
        context.artifacts[f"{self.step_id}.results"] = targets
        context.results = targets
        return context


__all__ = [
    "AttachStep",
    "BoundModelStep",
    "CompositionStep",
    "CropStep",
    "FilterStep",
    "ModelStep",
    "RegionInput",
    "crop_region",
]

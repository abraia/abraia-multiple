"""Shared pipeline renderers for model-specific visual results."""

import copy


class AnomalyPipelineRenderer:
    """Blend calibrated anomaly maps beneath the model's image-level result."""

    def __init__(self, renderer, model):
        self.renderer = renderer
        self.model = model

    def __call__(self, context):
        anomaly_map = self.model.last_normalized_anomaly_map
        if anomaly_map is None:
            return (
                self.renderer(context)
                if self.renderer
                else context.frame.copy()
            )

        from ..inference.models.anomaly import render_anomaly_heatmap

        blended = render_anomaly_heatmap(context.frame, anomaly_map)
        render_context = copy.copy(context)
        render_context.frame = blended
        render_context.results = [
            {
                "label": result["label"],
                "score": result.get("score"),
                "class_id": result.get("class_id", 0),
            }
            for result in context.results or ()
            if isinstance(result, dict)
            and result.get("label") in ("normal", "anomaly")
        ]
        return (
            self.renderer(render_context)
            if self.renderer
            else blended
        )


__all__ = ["AnomalyPipelineRenderer"]

"""Canonical result records published by pipeline frames."""

import math
from collections.abc import Mapping
from numbers import Integral, Real

from .geometry import add_calibrated_measurements


def _first_present(*values, default=None):
    return next((
        value for value in values
        if value is not None and not (isinstance(value, str) and not value)
    ), default)


def set_anomaly_decisions(records):
    """Add decisions to records from an anomaly model."""
    for record in records:
        if not isinstance(record, dict):
            continue
        prediction = record.get("prediction") or record.get("label") or ""
        decision = {"normal": "PASS", "anomaly": "FAIL"}.get(str(prediction).casefold())
        if decision:
            record.setdefault("decision_result", decision)


def result_records(values, calibration=None):
    """Add table-ready fields to full pipeline result dictionaries."""
    calibration = calibration or {}
    records = []
    for value in values or ():
        if not isinstance(value, Mapping):
            continue
        record = dict(value)
        classification = value.get("classification")
        regression = value.get("regression")
        text = value.get("text")
        score = _first_present(*(
            value.get(name) for name in ("text_score", "score", "confidence")
        ))
        if isinstance(classification, Mapping):
            row_classification = _first_present(*(
                classification.get(name) for name in ("label", "text", "class_id")
            ))
            prediction = _first_present(
                value.get("label"), text, row_classification, default="—"
            )
            classification_score = classification.get("score")
        elif isinstance(regression, Mapping):
            prediction = _first_present(
                value.get("label"), regression.get("prediction"), default="—"
            )
            row_classification = None
            classification_score = None
        elif _first_present(text) is not None:
            prediction = text
            row_classification = None
            classification_score = None
        elif _first_present(value.get("label")) is not None:
            prediction = value["label"]
            row_classification = None
            classification_score = None
        elif "similarity" in value:
            prediction = _first_present(value["similarity"], default="—")
            row_classification = None
            classification_score = None
        else:
            prediction = _first_present(value.get("value"), default="—")
            row_classification = None
            classification_score = None

        if record.get("prediction") is None:
            record["prediction"] = prediction
        if score is not None and record.get("score") is None:
            record["score"] = score
        if row_classification is not None:
            record.setdefault("classification_label", row_classification)
        if classification_score is not None:
            record.setdefault("classification_score", classification_score)
        if isinstance(regression, Mapping):
            if regression.get("prediction") is not None:
                record.setdefault("regression_value", regression["prediction"])
            if regression.get("target") is not None:
                record.setdefault("regression_target", regression["target"])
            if regression.get("score") is not None:
                record.setdefault("regression_score", regression["score"])
        for field in ("ocr", "recognition", "result"):
            output = value.get(field)
            if isinstance(output, Mapping):
                for name in ("text", "label", "prediction", "score", "text_score"):
                    if output.get(name) is not None:
                        column = f"{field}_{'score' if name == 'text_score' else name}"
                        record.setdefault(column, output[name])
        measurements = value.get("measurements")
        if isinstance(measurements, Mapping):
            calibrated = add_calibrated_measurements(dict(measurements), calibration)
            record.update(calibrated)

        anomaly = value.get("anomaly")
        if isinstance(anomaly, Mapping):
            anomaly_prediction = anomaly.get("prediction", "review")
            record.setdefault("anomaly_prediction", anomaly_prediction)
            record.setdefault("anomaly_score", anomaly.get("score"))
            decision = {"normal": "PASS", "anomaly": "FAIL"}.get(
                str(anomaly_prediction).casefold()
            )
            if decision:
                record.setdefault("decision_result", decision)
        records.append(record)
    return records


_MEASUREMENT_FIELDS = {
    "aspect_ratio", "orientation_deg", "compactness", "solidity", "time_in_region_s",
}
_MEASUREMENT_FIELDS.update(
    f"{field}_{unit}"
    for field in (
        "centroid", "bounds", "width", "height", "perimeter",
        "major_axis", "minor_axis",
    )
    for unit in ("px", "mm", "cm", "µm")
)
_MEASUREMENT_FIELDS.update(("area_px", "area_mm2", "area_cm2", "area_µm2"))
_TABLE_EXCLUDED_FIELDS = {
    "id", "parent_id", "item_id", "box", "source_box", "points", "polygon",
    "mask", "item", "features", "details",
}
_TABLE_FIELD_ALIASES = {
    "label": "prediction", "text": "prediction", "value": "prediction",
    "similarity": "prediction", "confidence": "score", "text_score": "score",
}


def _result_decimal_places(name, value):
    if name == "anomaly_score":
        return 2
    magnitude = abs(value)
    if name in _MEASUREMENT_FIELDS and math.isfinite(magnitude) and magnitude > 0:
        return max(0, 2 - math.floor(math.log10(magnitude)))
    return 3


def round_result_value(name, value):
    """Round a numeric rule input to the precision used by the results table."""
    if isinstance(value, Integral):
        return int(value)
    value = float(value)
    return round(value, _result_decimal_places(name, value))


def _table_number(name, value):
    if isinstance(value, Integral) and name != "anomaly_score":
        return str(value)
    value = float(value)
    decimals = _result_decimal_places(name, value)
    text = f"{value:.{decimals}f}"
    if name == "anomaly_score":
        return text
    return text.rstrip("0").rstrip(".") if "." in text else text


def table_result_records(records, *, limit=None):
    """Export compact records with display precision, omitting internal IDs."""
    rows = []
    for record in records[:limit]:
        row = {}
        for name, value in record.items():
            alias = _TABLE_FIELD_ALIASES.get(name)
            if name in _TABLE_EXCLUDED_FIELDS or (
                alias and record.get(alias) is not None
            ):
                continue
            if isinstance(value, str):
                row[name] = value
            elif isinstance(value, Integral) and name != "anomaly_score":
                row[name] = value if isinstance(value, bool) else int(value)
            elif isinstance(value, Real):
                row[name] = _table_number(name, value)
            elif isinstance(value, (list, tuple)) and len(value) <= 4 and all(
                isinstance(part, Real) for part in value
            ):
                parts = ", ".join(_table_number(name, part) for part in value)
                row[name] = f"({parts})"
        rows.append(row)
    return rows

from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from abraia import cli as cli_module
from abraia.runtime import Pipeline


@pytest.mark.parametrize(
    "source, expected_source, expected_type",
    [(None, 0, "camera"), ("input.mp4", "input.mp4", "video"), ("faces", "faces", "image")],
)
def test_project_run_applies_source_and_output_overrides(
    tmp_path, monkeypatch, source, expected_source, expected_type
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "faces").mkdir()
    config = {
        "version": 2,
        "source": {"type": "camera", "src": 0},
        "steps": [{"type": "model", "model": {"kind": "yolov8"}}],
    }
    load_config = MagicMock(return_value=config)
    build_pipeline = MagicMock()
    monkeypatch.setattr(cli_module.abraia, "load_json", load_config)
    monkeypatch.setattr(Pipeline, "from_dict", build_pipeline)
    args = ["run", "project"]
    if source is not None:
        args.append(source)
    args.extend(["--output", "result.avi", "--accelerator", "cpu"])

    result = CliRunner().invoke(cli_module.cli, args)

    assert result.exit_code == 0, result.output
    load_config.assert_called_once_with("project/pipeline.json")
    assert config["source"] == {"type": expected_type, "src": expected_source}
    assert config["display"]["dest"] == "result.avi"
    build_pipeline.assert_called_once_with(config, accelerator="cpu")
    build_pipeline.return_value.__enter__.return_value.run.assert_called_once_with()
    build_pipeline.return_value.__exit__.assert_called_once()


@pytest.mark.parametrize(
    "demo, runner", [("detect", "monitor_objects"), ("faces", "track_faces")]
)
def test_demo_errors_use_cli_error_format(monkeypatch, demo, runner):
    def fail(*_args, **_kwargs):
        raise RuntimeError("No compatible device")

    monkeypatch.setattr(f"abraia.demo.{runner}", fail)

    result = CliRunner().invoke(cli_module.cli, ["run", "demo", demo])

    assert result.exit_code == 1
    assert "Error: No compatible device" in result.output

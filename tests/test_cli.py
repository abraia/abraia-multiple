from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from abraia import config
from abraia.cli import cli
from abraia import cli as cli_module


def test_curate_requires_confirmation_before_applying_changes():
    captured = {}

    def fake_curate(**kwargs):
        captured["confirmed"] = kwargs["confirm"](["project/image.jpg"])
        return {"deleted": []}

    def fake_load_dataset(project):
        captured["project"] = project
        return SimpleNamespace(curate=fake_curate)

    with patch(
        "abraia.dataset.load_dataset",
        side_effect=fake_load_dataset,
    ):
        result = CliRunner().invoke(
            cli,
            ["curate", "project", "--apply"],
            input="n\n",
        )

    assert result.exit_code == 0
    assert captured == {"project": "project", "confirmed": False}
    assert "Delete 1 recommended remote file(s)?" in result.output


def test_metadata_remove_does_not_reload_deleted_metadata(monkeypatch):
    removed = []
    monkeypatch.setattr(
        cli_module.abraia,
        "remove_metadata",
        lambda path: removed.append(path),
    )
    monkeypatch.setattr(
        cli_module.abraia,
        "load_metadata",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("deleted metadata should not be loaded")
        ),
    )

    result = CliRunner().invoke(
        cli,
        ["files", "metadata", "--remove", "project/image.jpg"],
    )

    assert result.exit_code == 0
    assert removed == ["project/image.jpg"]
    assert "Removed metadata for project/image.jpg" in result.output


def test_compile_accepts_thresholds_as_arguments(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch(
        "abraia.training.artifacts.list_model_records",
        return_value=[{
            "name": "yolov8m_v2.onnx",
            "task": "detection",
            "classes": ["object"],
            "backend": "ultralytics",
            "modelSize": "medium",
            "inputShape": [1, 3, 512, 512],
            "artifacts": {"checkpoint": "project/yolov8m_v2.pt"},
        }],
    ), patch.object(
        cli_module.abraia,
        "download_file",
        return_value="/cache/yolov8m_v2.pt",
    ) as download_file, patch(
        "abraia.training.orchestration._compile_checkpoint",
        return_value={"hef": "project/model.hef"},
    ) as compile_checkpoint:
        result = CliRunner().invoke(
            cli,
            [
                "compile", "project", "--device", "hailo8l",
                "--conf", "0.4", "--iou", "0.3",
            ],
        )

    assert result.exit_code == 0, result.output
    download_file.assert_called_once_with("project/yolov8m_v2.pt")
    assert compile_checkpoint.call_args.args[1:3] == (
        "detection",
        ["object"],
    )
    assert compile_checkpoint.call_args.args[3] == "/cache/yolov8m_v2.pt"
    assert compile_checkpoint.call_args.args[4] == "ultralytics"
    assert compile_checkpoint.call_args.kwargs["model_size"] == "medium"
    assert compile_checkpoint.call_args.kwargs["imgsz"] == 512
    assert compile_checkpoint.call_args.kwargs["device"] == "hailo8l"
    assert compile_checkpoint.call_args.kwargs["conf"] == 0.4
    assert compile_checkpoint.call_args.kwargs["iou"] == 0.3
    assert compile_checkpoint.call_args.kwargs["config_path"] is None
    assert compile_checkpoint.call_args.kwargs["model_version"] == 2


def test_train_passes_image_size_to_trainer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dataset = SimpleNamespace(task="detection", classes=["object"])
    trainer = SimpleNamespace(
        train=lambda epochs: None,
        test=lambda: {},
        save=lambda: None,
    )
    with patch(
        "abraia.dataset.load_dataset", return_value=dataset
    ), patch("abraia.dataset.preparation.prepare_dataset"), patch(
        "abraia.training.orchestration.ModelTrainer", return_value=trainer
    ) as model_trainer:
        result = CliRunner().invoke(
            cli,
            ["train", "project", "--imgsz", "512", "--model-size", "large"],
        )

    assert result.exit_code == 0, result.output
    assert model_trainer.call_args.kwargs["imgsz"] == 512
    assert model_trainer.call_args.kwargs["model_size"] == "large"
    assert not (tmp_path / "project" / ".abraia" / "training.json").exists()


def test_custom_run_falls_back_to_best_model_and_closes_pipeline(monkeypatch):
    from abraia.client import APIError

    pipeline_config = {
        "source": {"type": "video", "src": "camera.mp4"},
        "steps": [{"model": {"kind": "yolov8", "uri": "builtin.onnx"}}],
    }
    calls = {}

    class FakePipeline:
        def __init__(self, config):
            self.config = config
            self.closed = False
            calls["pipeline"] = self

        @classmethod
        def from_dict(cls, config, accelerator):
            calls["config"] = config
            calls["accelerator"] = accelerator
            return cls(config)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.closed = True

        def run(self):
            calls["ran"] = True

    monkeypatch.setattr(cli_module.abraia, "userid", "user")
    with patch.object(
        cli_module.abraia,
        "load_json",
        side_effect=APIError("Not found", code=404),
    ):
        with patch(
            "abraia.training.artifacts.list_model_records",
            return_value=[
                {"name": "model.onnx", "task": "segmentation", "kind": "rtmdet"}
            ],
        ) as list_models:
            with patch("abraia.demo.resolve_pipeline", return_value=pipeline_config):
                with patch("abraia.runtime.Pipeline", FakePipeline):
                    result = CliRunner().invoke(
                        cli,
                        ["run", "project", "detect", "image.jpg"],
                    )

    assert result.exit_code == 0, result.output
    list_models.assert_called_once_with(
        "project", client=cli_module.abraia, sort_by_accuracy=True
    )
    assert calls["config"]["steps"][0]["model"] == {
        "kind": "rtmdet",
        "task": "segmentation",
        "uri": "user/project/model.onnx",
    }
    assert calls["config"]["source"] == {"type": "image", "src": "image.jpg"}
    assert calls["accelerator"] == "auto"
    assert calls["ran"] is True
    assert calls["pipeline"].closed is True


def test_configure_prompts_for_and_saves_only_the_api_key(monkeypatch, tmp_path):
    encoded_key = config.base64encode("configured-user:secret")
    config_path = tmp_path / "abraia"
    monkeypatch.setattr(config, "CONFIG_FILE", str(config_path))

    result = CliRunner().invoke(cli, ["configure"], input=f"{encoded_key}\n")

    assert result.exit_code == 0
    assert "Abraia Key" in result.output
    assert config_path.read_text(encoding="utf-8") == f"abraia_key: {encoded_key}\n"

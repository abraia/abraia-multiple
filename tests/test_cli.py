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


def test_train_passes_image_size_to_training_service(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch("abraia.training.Training") as training_cls:
        training_cls.return_value.train.return_value = {"stats": {"acc": 0.9}}
        result = CliRunner().invoke(
            cli,
            ["train", "project", "--imgsz", "512", "--model-size", "large"],
        )

    assert result.exit_code == 0, result.output
    training_cls.return_value.train.assert_called_once()
    assert training_cls.return_value.train.call_args.kwargs["image_size"] == 512
    assert training_cls.return_value.train.call_args.kwargs["model_size"] == "large"
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

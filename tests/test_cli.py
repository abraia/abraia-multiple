from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from abraia.cli import cli
from abraia import cli as cli_module


def test_curate_requires_confirmation_before_applying_changes():
    captured = {}

    def fake_curate_dataset(project, **kwargs):
        captured["project"] = project
        captured["confirmed"] = kwargs["confirm"](["project/image.jpg"])
        return SimpleNamespace(to_dict=lambda: {"deleted": []})

    with patch("abraia.training.curate_dataset", side_effect=fake_curate_dataset):
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


def test_custom_run_closes_model(monkeypatch):
    class FakeModel:
        instances = []

        def __init__(self, path):
            self.path = path
            self.closed = False
            self.__class__.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.closed = True

        def run(self, image):
            return []

    monkeypatch.setattr(cli_module.abraia, "userid", "user")
    with patch("abraia.training.list_models", return_value=["model.onnx"]):
        with patch(
            "abraia.inference.models.detection.Model",
            FakeModel,
        ):
            with patch("abraia.cli.process_media") as process_media:
                result = CliRunner().invoke(
                    cli,
                    ["run", "project", "detect", "image.jpg"],
                )

    assert result.exit_code == 0
    assert process_media.called
    assert FakeModel.instances[0].path == "user/project/model.onnx"
    assert FakeModel.instances[0].closed is True


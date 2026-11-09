from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from click.testing import CliRunner
from click.shell_completion import BashComplete
import pytest

from abraia import config
from abraia.cli import cli
from abraia import cli as cli_module
from abraia.runtime import Pipeline


def test_cli_exposes_sdk_operations_and_inference():
    assert set(cli.commands) == {
        "configure", "list", "upload", "download", "remove",
        "run", "search", "discover",
    }
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "upload" in result.output


@pytest.mark.parametrize(
    "shell, relative_path, expected",
    [
        ("bash", ".bashrc", "_ABRAIA_COMPLETE=bash_source abraia"),
        ("zsh", ".zshrc", "_ABRAIA_COMPLETE=zsh_source abraia"),
        (
            "fish",
            ".config/fish/completions/abraia.fish",
            "complete --no-files --command abraia",
        ),
    ],
)
def test_completion_install_configures_shell_once(
    tmp_path, monkeypatch, shell, relative_path, expected
):
    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / relative_path

    first = CliRunner().invoke(
        cli, ["configure", "--completion", "--shell", shell]
    )
    second = CliRunner().invoke(
        cli, ["configure", "--completion", "--shell", shell]
    )

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    content = path.read_text(encoding="utf-8")
    assert expected in content
    assert content.count(expected) == 1
    assert "already installed" in second.output


def test_completion_install_detects_shell_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHELL", "/bin/zsh")

    result = CliRunner().invoke(cli, ["configure", "--completion"])

    assert result.exit_code == 0, result.output
    assert (tmp_path / ".zshrc").is_file()


def test_run_help_shows_demo_and_saved_pipeline_examples():
    result = CliRunner().invoke(cli, ["run", "--help"])

    assert result.exit_code == 0
    assert "PROJECT [DEMO_OR_SOURCE] [SOURCE]" in result.output
    assert "abraia run demo detect 0" in result.output
    assert "abraia run my-project input.mp4" in result.output


def test_discover_cameras_command_lists_sources(monkeypatch):
    from abraia import runtime

    monkeypatch.setattr(
        runtime,
        "discover_cameras",
        lambda: [{"source": "1", "label": "Camera 1"}],
    )

    result = CliRunner().invoke(cli, ["discover", "cameras"])

    assert result.exit_code == 0, result.output
    assert "1  Camera 1" in result.output
    assert "abraia run demo detect SOURCE" in result.output


def test_discover_streams_command_lists_run_example(monkeypatch):
    from abraia import runtime

    monkeypatch.setattr(
        runtime,
        "discover_streams",
        lambda: ["rtsp://camera-address/stream"],
    )

    result = CliRunner().invoke(cli, ["discover", "streams"])

    assert result.exit_code == 0, result.output
    assert "rtsp://camera-address/stream" in result.output
    assert "abraia run demo detect STREAM_URL" in result.output


@pytest.mark.parametrize(
    "source, expected_type",
    [
        ("1", "camera"),
        ("rtsp://camera-address/stream", "stream"),
        ("input.mp4", "video"),
    ],
)
def test_demo_source_override_infers_camera_and_stream_types(source, expected_type):
    from abraia.demo import _prepare_source

    pipeline = {"source": {"type": "video", "src": "sample.mp4"}}

    _prepare_source(pipeline, source)

    assert pipeline["source"]["type"] == expected_type


def test_demo_source_override_keeps_pi_camera_type():
    from abraia.demo import _prepare_source

    pipeline = {"source": {"type": "rpi_camera", "src": 0}}

    _prepare_source(pipeline, "0")

    assert pipeline["source"]["type"] == "rpi_camera"


@pytest.mark.parametrize("args", [["demo"], ["demo", "list"]])
def test_run_demo_lists_workflows_and_model_presets(args):
    result = CliRunner().invoke(cli, ["run", *args])

    assert result.exit_code == 0, result.output
    assert "Use-case demos" in result.output
    assert "apple" in result.output
    assert "Model demos" in result.output
    assert "abraia run demo apple --model yolov8n-seg" in result.output


def test_run_demo_selects_a_model_variant_explicitly(monkeypatch):
    from abraia import demo as demo_module

    calls = []
    monkeypatch.setattr(
        demo_module,
        "monitor_objects",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    result = CliRunner().invoke(
        cli,
        [
            "run", "demo", "apple", "input.mp4",
            "--model", "yolov8n-seg",
            "--accelerator", "cpu", "--output", "out.avi",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [(
        ("input.mp4", "apple"),
        {"accelerator": "cpu", "output": "out.avi", "model": "yolov8n-seg"},
    )]


@pytest.mark.parametrize(
    "args, expected",
    [
        (["run", "demo", "apples"], "Unknown demo 'apples'"),
        (["run", "demo", "people", "--model", "yolo9"], "Unknown model variant 'yolo9'"),
    ],
)
def test_run_demo_rejects_aliases(args, expected):
    result = CliRunner().invoke(cli, args)

    assert result.exit_code == 1
    assert expected in result.output


def test_run_command_completes_demo_names_and_compatible_model_variants():
    context = SimpleNamespace(params={"project": "demo", "demo_or_source": "apple"})

    demo_names = cli_module._complete_run_demo(context, None, "app")
    model_names = cli_module._complete_demo_model(context, None, "rfdetr-seg")

    assert [item.value for item in demo_names] == ["apple"]
    assert [item.value for item in model_names] == [
        "rfdetr-seg-nano", "rfdetr-seg-small", "rfdetr-seg-medium"
    ]


def test_bash_completion_completes_models_after_option():
    completion = BashComplete(cli, {}, "abraia", "_ABRAIA_COMPLETE")

    items = completion.get_completions(
        ["run", "demo", "apple", "--model"],
        "yolov8n-seg",
    )

    assert [item.value for item in items] == ["yolov8n-seg"]


@pytest.mark.parametrize("command", [
    "files", "create", "annotate", "curate", "train", "compile",
])
def test_cli_rejects_removed_workflow_commands_and_files_group(command):
    result = CliRunner().invoke(cli, [command])

    assert result.exit_code == 2
    assert f"No such command '{command}'" in result.output


def test_top_level_list_shows_remote_files_and_folders(monkeypatch):
    paths = []

    def list_files(path):
        paths.append(path)
        return (
            [{"name": "image.jpg", "size": 12, "date": "2026-10-23"}],
            [{"name": "models"}],
        )

    monkeypatch.setattr(cli_module.abraia, "list_files", list_files)

    result = CliRunner().invoke(cli, ["list", "project/"])

    assert result.exit_code == 0, result.output
    assert paths == ["project/"]
    assert "image.jpg" in result.output
    assert "models/" in result.output


def test_top_level_remove_keeps_confirmation(monkeypatch):
    removed = []
    monkeypatch.setattr(cli_module.abraia, "list_files", lambda path: (
        [{"name": "image.jpg", "path": "project/image.jpg", "size": 12,
          "date": "2026-10-23"}], [],
    ))
    monkeypatch.setattr(cli_module.abraia, "remove_file", removed.append)

    result = CliRunner().invoke(cli, ["remove", "project/image.jpg"], input="n\n")

    assert result.exit_code == 0, result.output
    assert "Are you sure" in result.output
    assert removed == []


def test_configure_prompts_for_and_saves_only_the_api_key(monkeypatch, tmp_path):
    encoded_key = config.base64encode("configured-user:secret")
    config_path = tmp_path / "abraia"
    monkeypatch.setattr(config, "CONFIG_FILE", str(config_path))

    result = CliRunner().invoke(cli, ["configure"], input=f"{encoded_key}\n")

    assert result.exit_code == 0
    assert "Abraia Key" in result.output
    assert config_path.read_text(encoding="utf-8") == f"abraia_key: {encoded_key}\n"


@pytest.mark.parametrize(
    "source, expected_source, expected_type",
    [
        (None, 0, "camera"),
        ("input.mp4", "input.mp4", "video"),
        ("faces", "faces", "image"),
    ],
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

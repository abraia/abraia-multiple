import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

from click.testing import CliRunner
from click.shell_completion import BashComplete
import numpy as np
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


def test_run_help_shows_demo_and_saved_pipeline_arguments():
    result = CliRunner().invoke(cli, ["run", "--help"])

    assert result.exit_code == 0
    assert "PROJECT [DEMO_OR_SOURCE] [SOURCE]" in result.output
    assert "--accelerator" in result.output
    assert "--output" in result.output


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


def test_run_demo_lists_available_demos():
    result = CliRunner().invoke(cli, ["run", "demo", "list"])

    assert result.exit_code == 0, result.output
    assert "Available demos" in result.output
    assert "apple" in result.output
    assert "depth" in result.output
    for alias in ("yolo9", "rfdetr", "dfine", "rfdetr-seg", "dfine-seg"):
        assert alias not in result.output


def test_run_demo_without_name_loads_public_detect_demo(monkeypatch, public_demo_configs):
    from abraia import demo as demo_module

    config = public_demo_configs["detect"]
    fetch = MagicMock(return_value=BytesIO(json.dumps(config).encode()))
    build_pipeline = MagicMock()
    monkeypatch.setattr(demo_module, "load_url", fetch)
    monkeypatch.setattr(Pipeline, "from_dict", build_pipeline)

    result = CliRunner().invoke(cli, ["run", "demo"])

    assert result.exit_code == 0, result.output
    fetch.assert_called_once_with(
        "https://api.abraia.me/files/multiple/demos/detect.json", timeout=ANY
    )
    pipeline_config = build_pipeline.call_args.args[0]
    assert pipeline_config["source"] == {
        "type": "camera", "src": 0, "resolution": [1280, 720],
    }
    assert pipeline_config["steps"] == config["steps"]


def test_run_demo_fails_when_public_detect_config_cannot_be_fetched(monkeypatch):
    monkeypatch.setattr("abraia.demo.load_url", lambda *args, **kwargs: None)
    build = MagicMock()
    monkeypatch.setattr(Pipeline, "from_dict", build)

    result = CliRunner().invoke(cli, ["run", "demo"])

    assert result.exit_code == 1
    assert "Could not download public demo 'detect'" in result.output
    build.assert_not_called()


def test_run_demo_selects_a_model_variant_explicitly(monkeypatch, public_demo_configs):
    from abraia import demo as demo_module

    payload = json.dumps(public_demo_configs["apple"]).encode()
    fetch = MagicMock(return_value=BytesIO(payload))
    build_pipeline = MagicMock()
    sample = MagicMock()
    monkeypatch.setattr(demo_module, "load_url", fetch)
    monkeypatch.setattr("abraia.utils.remote.download_file", sample)
    monkeypatch.setattr(Pipeline, "from_dict", build_pipeline)

    result = CliRunner().invoke(
        cli,
        [
            "run", "demo", "apple", "input.mp4",
            "--model", "yolov8m-seg",
            "--accelerator", "cpu", "--output", "out.avi",
        ],
    )

    assert result.exit_code == 0, result.output
    fetch.assert_called_once_with(
        "https://api.abraia.me/files/multiple/demos/apple.json", timeout=ANY
    )
    sample.assert_not_called()
    pipeline_config = build_pipeline.call_args.args[0]
    assert pipeline_config["source"] == {"src": "input.mp4", "type": "video"}
    assert pipeline_config["display"]["dest"] == "out.avi"
    assert pipeline_config["steps"][0]["model"]["uri"].endswith("/yolov8m-seg.onnx")
    assert build_pipeline.call_args.kwargs["accelerator"] == "cpu"


@pytest.mark.parametrize("variant,kind,uri", [
    ("yolo26n-depth", "yolo26_depth", "multiple/models/yolo26n-depth.onnx"),
    ("depth-anything-v3-small", "depth_anything", "multiple/models/depth_anything_v3_small.onnx"),
    ("depth-anything-v2-small", "depth_anything_v2", "multiple/models/depth_anything_v2_small.onnx"),
])
def test_depth_demo_cli_uses_the_public_pipeline(
    monkeypatch, public_demo_configs, variant, kind, uri,
):
    from abraia import demo as demo_module

    context = SimpleNamespace(params={"project": "demo", "demo_or_source": "depth"})
    assert [item.value for item in cli_module._complete_run_demo(context, None, "dep")] == [
        "depth",
    ]
    assert [item.value for item in cli_module._complete_demo_model(context, None, "")] == [
        "yolo26n-depth",
        "depth-anything-v3-small",
        "depth-anything-v2-small",
    ]

    config = public_demo_configs["depth"]
    config["steps"][0]["model"].pop("uri")
    payload = json.dumps(config).encode()
    fetch = MagicMock(side_effect=lambda *args, **kwargs: BytesIO(payload))
    build = MagicMock()
    monkeypatch.setattr(demo_module, "load_url", fetch)
    monkeypatch.setattr(Pipeline, "from_dict", build)

    runner = CliRunner()
    result = runner.invoke(cli, ["run", "demo", "depth", "--accelerator", "cpu"])
    assert result.exit_code == 0, result.output
    assert build.call_args.args[0]["source"] == {
        "type": "camera", "src": 0, "resolution": [1280, 720],
    }
    assert build.call_args.args[0]["steps"] == config["steps"]
    assert "renderer_factory" not in build.call_args.kwargs

    result = runner.invoke(cli, [
        "run", "demo", "depth", "input.mp4", "--accelerator", "cpu",
        "--model", variant, "--output", "depth.avi",
    ])
    assert result.exit_code == 0, result.output
    assert [call.args[0] for call in fetch.call_args_list] == [
        "https://api.abraia.me/files/multiple/demos/depth.json",
    ] * 2
    selected = build.call_args.args[0]
    assert selected["source"] == {"type": "video", "src": "input.mp4"}
    assert selected["display"]["dest"] == "depth.avi"
    assert selected["steps"][0]["model"]["kind"] == kind
    assert selected["steps"][0]["model"]["uri"] == uri


@pytest.mark.parametrize("units", ["m", "relative"])
def test_runtime_depth_renderer_overlays_colors_without_changing_the_map(units):
    from abraia.runtime.pipeline import _make_depth_renderer

    depth = np.tile([0, -1, np.nan, np.inf, 1, 5, 10, 20], (80, 40)).astype(np.float32)
    original = depth.copy()
    context = SimpleNamespace(
        frame=np.full((*depth.shape, 3), 222, dtype=np.uint8),
        results=[{"depth_units": units, "depth_map": depth}],
    )
    render = _make_depth_renderer(lambda context: context.frame)
    image = render(context)
    assert image.shape == context.frame.shape
    assert image.dtype == np.uint8
    np.testing.assert_array_equal(image[-1, :4], context.frame[-1, :4])
    assert not np.array_equal(image[-1, 4], context.frame[-1, 4])
    assert image[-1, 4].sum() > image[-1, 5].sum() > image[-1, 6].sum()
    if units == "m":
        np.testing.assert_array_equal(image[-1, 6], image[-1, 7])
    else:
        assert image[-1, 6].sum() > image[-1, 7].sum()
    np.testing.assert_array_equal(depth, original)
    assert np.all(context.frame == 222)
    context.results.clear()
    assert render(context) is context.frame


def test_explicit_hailo_does_not_launch_an_onnx_only_demo(monkeypatch, public_demo_configs):
    monkeypatch.setattr("abraia.demo.load_public_pipeline", lambda name: public_demo_configs[name])
    monkeypatch.setattr("abraia.runtime.pipeline.create_frame_source", MagicMock())
    model = MagicMock()
    monkeypatch.setattr("abraia.inference.models.depth.DepthModel", model)

    result = CliRunner().invoke(cli, ["run", "demo", "depth", "--accelerator", "hailo"])

    assert result.exit_code == 1
    assert "Hailo required for model 'yolo26_depth'" in result.output
    model.assert_not_called()


def test_public_demo_reloads_json_and_downloads_sample_without_account(
    monkeypatch, public_demo_configs
):
    from abraia import demo as demo_module

    sources = ["multiple/videos/first.mp4", "multiple/videos/second.mp4"]
    configurations = [
        dict(public_demo_configs["apple"], source={"type": "video", "src": source})
        for source in sources
    ]
    fetch = MagicMock(side_effect=[
        BytesIO(json.dumps(configuration).encode()) for configuration in configurations
    ])
    download = MagicMock(side_effect=["/cached/first.mp4", "/cached/second.mp4"])
    build_pipeline = MagicMock()
    private_load = MagicMock()
    monkeypatch.setattr(demo_module, "load_url", fetch)
    monkeypatch.setattr("abraia.utils.remote.download_file", download)
    monkeypatch.setattr(Pipeline, "from_dict", build_pipeline)
    monkeypatch.setattr(cli_module.abraia, "load_json", private_load)

    for _ in configurations:
        result = CliRunner().invoke(cli, ["run", "demo", "apple"])
        assert result.exit_code == 0, result.output

    assert [call.args[0] for call in fetch.call_args_list] == [
        "https://api.abraia.me/files/multiple/demos/apple.json",
    ] * 2
    assert [call.args[0] for call in download.call_args_list] == sources
    assert [call.args[0]["source"]["src"] for call in build_pipeline.call_args_list] == [
        "/cached/first.mp4", "/cached/second.mp4",
    ]
    assert all(call.kwargs["accelerator"] == "onnx" for call in build_pipeline.call_args_list)
    private_load.assert_not_called()


@pytest.mark.parametrize(
    "args, expected",
    [
        (["run", "demo", "apples"], "Unknown demo 'apples'"),
        (["run", "demo", "people", "--model", "yolo9"], "Unknown model variant 'yolo9'"),
    ],
)
def test_run_demo_rejects_aliases(monkeypatch, public_demo_configs, args, expected):
    from abraia import demo as demo_module

    monkeypatch.setattr(
        demo_module, "load_public_pipeline", lambda name: public_demo_configs[name]
    )
    result = CliRunner().invoke(cli, args)

    assert result.exit_code == 1
    assert expected in result.output


def test_run_command_completes_demo_names_and_compatible_model_variants():
    context = SimpleNamespace(params={"project": "demo", "demo_or_source": "apple"})

    demo_names = cli_module._complete_run_demo(context, None, "app")
    model_names = cli_module._complete_demo_model(context, None, "rfdetr-seg")
    targets = cli_module._complete_run_project(context, None, "dem")

    assert [item.value for item in demo_names] == ["apple"]
    assert [item.value for item in targets] == ["demo"]
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
    build_pipeline.assert_called_once_with(
        config, accelerator="cpu", on_frame=ANY
    )
    build_pipeline.return_value.__enter__.return_value.run.assert_called_once_with(
        hold_image=True
    )
    build_pipeline.return_value.__exit__.assert_called_once()


@pytest.mark.parametrize(
    "source_config, override, remote_path, expected_source",
    [
        (
            {"type": "image", "src": "fuji-conveyor-anomaly-severe.png"},
            None, "apples/fuji-conveyor-anomaly-severe.png", None,
        ),
        ({"src": "apples/clips/input.mp4"}, None, "apples/clips/input.mp4", None),
        ({"type": "image", "src": "default.png"}, "local.png", None, "local.png"),
        ({"type": "camera", "src": 0}, None, None, 0),
        ({"type": "genicam", "src": "camera.jpg"}, None, None, "camera.jpg"),
        ({"type": "video", "src": "https://example.org/input.mp4"},
         None, None, "https://example.org/input.mp4"),
        ({"type": "image", "src": "/local/input.png"}, None, None, "/local/input.png"),
    ],
)
def test_project_run_downloads_only_saved_dataset_files(
    tmp_path, monkeypatch, source_config, override, remote_path, expected_source
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "fuji-conveyor-anomaly-severe.png").touch()
    config = {"version": 2, "source": dict(source_config)}
    local_path = str(tmp_path / "cached" / "input")
    load_config = MagicMock(return_value=config)
    materialize = MagicMock(return_value=local_path)
    build_pipeline = MagicMock()
    monkeypatch.setattr(cli_module.abraia, "load_json", load_config)
    monkeypatch.setattr(cli_module.abraia, "download_file", materialize)
    monkeypatch.setattr(Pipeline, "from_dict", build_pipeline)

    args = ["run", "apples"] + ([override] if override is not None else [])
    result = CliRunner().invoke(cli_module.cli, args)

    assert result.exit_code == 0, result.output
    load_config.assert_called_once_with("apples/pipeline.json")
    if remote_path:
        materialize.assert_called_once_with(remote_path)
        expected_source = local_path
    else:
        materialize.assert_not_called()
    assert build_pipeline.call_args[0][0]["source"]["src"] == expected_source
    build_pipeline.return_value.__enter__.return_value.run.assert_called_once_with(
        hold_image=True
    )


def test_project_run_prints_compact_result_table(monkeypatch):
    config = {"version": 2, "source": {"type": "camera", "src": 0}}
    build_pipeline = MagicMock()

    def run(**_kwargs):
        report = build_pipeline.call_args.kwargs["on_frame"]
        report(SimpleNamespace(frame_index=0, results=[
            {
                "prediction": "apple", "anomaly_score": 0.917,
                "box": np.array([0, 0, 10, 10]),
            },
        ]), 10)
        report(SimpleNamespace(frame_index=1, results=[]), 5)

    build_pipeline.return_value.__enter__.return_value.run.side_effect = run
    build_pipeline.return_value.__enter__.return_value.components = {}
    monkeypatch.setattr(cli_module.abraia, "load_json", lambda _path: config)
    monkeypatch.setattr(Pipeline, "from_dict", build_pipeline)

    result = CliRunner().invoke(cli_module.cli, ["run", "apples"])

    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        "Frame 0 (1 result)",
        "  prediction  anomaly_score",
        "  ----------  -------------",
        "  apple       0.92",
        "Frame 1 (0 results)",
    ]


def test_project_run_reports_missing_dataset_input(monkeypatch):
    config = {"version": 2, "source": {"type": "image", "src": "missing.png"}}
    download = MagicMock(side_effect=cli_module.APIError("Input file not found", 404))
    build_pipeline = MagicMock()
    monkeypatch.setattr(cli_module.abraia, "load_json", lambda path: config)
    monkeypatch.setattr(cli_module.abraia, "download_file", download)
    monkeypatch.setattr(Pipeline, "from_dict", build_pipeline)

    result = CliRunner().invoke(cli_module.cli, ["run", "apples"])

    assert result.exit_code == 1
    assert "Error: Input file not found" in result.output
    download.assert_called_once_with("apples/missing.png")
    build_pipeline.assert_not_called()


@pytest.mark.parametrize(
    "demo, runner", [("detect", "load_public_pipeline"), ("faces", "track_faces")]
)
def test_demo_errors_use_cli_error_format(monkeypatch, demo, runner):
    def fail(*_args, **_kwargs):
        raise RuntimeError("No compatible device")

    monkeypatch.setattr(f"abraia.demo.{runner}", fail)

    result = CliRunner().invoke(cli_module.cli, ["run", "demo", demo])

    assert result.exit_code == 1
    assert "Error: No compatible device" in result.output

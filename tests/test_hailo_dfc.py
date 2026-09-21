from unittest.mock import patch

from abraia.training import core


def test_hailo_dfc_wheels_match_target_generations():
    assert core.HAILO_DFC_WHEELS["hailo8"].startswith("multiple/hailo8/")
    assert core.HAILO_DFC_WHEELS["hailo8l"].endswith("3.33.1-py3-none-linux_x86_64.whl")
    assert core.HAILO_DFC_WHEELS["hailo10h"].startswith("multiple/hailo15/")
    assert core.HAILO_DFC_VERSIONS["hailo15h"] == "5.2.0"


def test_ensure_hailo_dfc_skips_matching_installation():
    with patch.object(core, "_supports_hailo_dfc", return_value=True), \
            patch.object(core, "_installed_dfc_version", return_value="3.33.1"), \
            patch.object(core, "_dfc_import_available", return_value=True), \
            patch("abraia.utils.remote.download_file") as download, \
            patch.object(core.subprocess, "run") as run:
        assert core.ensure_hailo_dfc("hailo8l") is False

    download.assert_not_called()
    run.assert_not_called()


def test_ensure_hailo_dfc_downloads_and_installs_missing_compiler():
    with patch.object(core, "_supports_hailo_dfc", return_value=True), \
            patch.object(core, "_installed_dfc_version", return_value=None), \
            patch.object(core, "_dfc_import_available", side_effect=[False, True]), \
            patch("abraia.utils.remote.download_file", return_value="/tmp/dfc.whl") as download, \
            patch.object(core.subprocess, "run") as run:
        run.return_value.returncode = 0

        assert core.ensure_hailo_dfc("hailo15h") is True

    download.assert_called_once_with(
        "multiple/hailo15/"
        "hailo_dataflow_compiler-5.2.0-py3-none-linux_x86_64.whl"
    )
    assert run.call_args.args[0] == [
        core.sys.executable,
        "-m",
        "pip",
        "install",
        "/tmp/dfc.whl",
    ]


def test_ensure_hailo_dfc_does_not_download_on_unsupported_host():
    with patch.object(core, "_supports_hailo_dfc", return_value=False), \
            patch("abraia.utils.remote.download_file") as download:
        assert core.ensure_hailo_dfc("hailo8") is False

    download.assert_not_called()

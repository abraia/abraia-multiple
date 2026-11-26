import sys
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from abraia.media.sources.genicam import GenICamSource, find_genicam_producers
from abraia.runtime.sources import create_source


def test_genicam_producer_is_found_from_environment_path(tmp_path, monkeypatch):
    producer = tmp_path / "vendor.cti"
    producer.touch()
    monkeypatch.setenv("GENICAM_GENTL64_PATH", str(tmp_path))

    assert find_genicam_producers() == [producer]


def test_genicam_numeric_serial_is_distinguished_from_device_index():
    device = SimpleNamespace(serial_number="123")

    assert GenICamSource._device_selector("123", [device]) == {
        "serial_number": "123"
    }
    assert GenICamSource._device_selector("1", [device]) == 1


def test_genicam_keeps_latest_frame_and_its_metadata(tmp_path, monkeypatch):
    acquirer = Mock(remote_device=SimpleNamespace(node_map=SimpleNamespace()))
    harvester = Mock(device_info_list=[SimpleNamespace(serial_number="123")])
    harvester.create.return_value = acquirer
    monkeypatch.setitem(sys.modules, "harvesters.core", SimpleNamespace(Harvester=lambda: harvester))
    monkeypatch.setattr(
        "abraia.media.sources.genicam.find_genicam_producers", lambda: [tmp_path / "vendor.cti"],
    )
    resume_capture = Event()

    def acquire(_source):
        yield {"frame": "first", "data": {"frame_id": 10}}
        assert resume_capture.wait(2)
        yield {"frame": "old", "data": {"frame_id": 11}}
        yield {"frame": "latest", "data": {"frame_id": 12}}

    monkeypatch.setattr(GenICamSource, "_read_frames", acquire)
    with GenICamSource("123") as source:
        frames = iter(source)
        try:
            assert next(frames) == {"frame": "first", "data": {"frame_id": 10}}
            assert source.last_capture_index == 0
            resume_capture.set()
            source._capture_thread.join(timeout=2)
            assert not source._capture_thread.is_alive()
            assert list(frames) == [{"frame": "latest", "data": {"frame_id": 12}}]
            assert source.last_capture_index == 2
        finally:
            resume_capture.set()
    acquirer.destroy.assert_called_once()


@pytest.mark.parametrize(
    "option",
    [
        {"producer": "/vendor/producer.cti"},
        {"selector": {"serial_number": "123"}},
        {"features": {"PixelFormat": "Mono8"}},
        {"timeout": 1.0},
    ],
)
def test_genicam_does_not_accept_pipeline_source_options(option):
    with pytest.raises(ValueError, match="GenICam source options are not supported"):
        create_source(
            "123",
            {
                "source_type": "genicam",
                "source_options": option,
            },
        )

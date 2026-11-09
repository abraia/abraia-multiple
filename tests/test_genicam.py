from types import SimpleNamespace

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

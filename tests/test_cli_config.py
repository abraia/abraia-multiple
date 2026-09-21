from click.testing import CliRunner

from abraia import config
from abraia.cli import cli


def test_configure_prompts_for_and_saves_only_the_api_key(monkeypatch, tmp_path):
    encoded_key = config.base64encode("configured-user:secret")
    config_path = tmp_path / "abraia"
    monkeypatch.setattr(config, "CONFIG_FILE", str(config_path))

    result = CliRunner().invoke(cli, ["configure"], input=f"{encoded_key}\n")

    assert result.exit_code == 0
    assert "Abraia Key" in result.output
    assert config_path.read_text(encoding="utf-8") == (
        f"abraia_key: {encoded_key}\n"
    )

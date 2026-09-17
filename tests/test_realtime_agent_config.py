import pytest

from realtime_agent.config import ConfigError, LiveKitConfig, load_config


def test_no_credentials_is_dry_run():
    cfg = LiveKitConfig.from_env(env={})
    assert cfg.is_configured is False


def test_full_credentials_is_configured():
    cfg = LiveKitConfig.from_env(
        env={
            "LIVEKIT_URL": "wss://example",
            "LIVEKIT_API_KEY": "key",
            "LIVEKIT_API_SECRET": "secret",
        }
    )
    assert cfg.is_configured is True
    assert cfg.require_configured() is cfg


def test_partial_credentials_is_hard_error():
    with pytest.raises(ConfigError):
        LiveKitConfig.from_env(env={"LIVEKIT_URL": "wss://example"})


def test_require_configured_raises_in_dry_run():
    with pytest.raises(ConfigError):
        LiveKitConfig.from_env(env={}).require_configured()


def test_room_and_identity_overridable():
    cfg = LiveKitConfig.from_env(
        env={"LIVEKIT_ROOM": "r1", "LIVEKIT_AGENT_IDENTITY": "bot-7"}
    )
    assert cfg.room_name == "r1"
    assert cfg.agent_identity == "bot-7"


def test_from_yaml_missing_path_is_hard_error():
    with pytest.raises(FileNotFoundError):
        LiveKitConfig.from_yaml("nope/missing.yaml")


def test_from_yaml_reads_room_and_identity(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("room_name: demo-room\nagent_identity: demo-agent\n")
    cfg = LiveKitConfig.from_yaml(p, env={})
    assert cfg.room_name == "demo-room"
    assert cfg.agent_identity == "demo-agent"
    assert cfg.is_configured is False  # no secrets in the file


def test_from_yaml_never_sources_secrets_from_the_file(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("url: wss://from-file\napi_key: file-key\napi_secret: file-secret\n")
    cfg = LiveKitConfig.from_yaml(
        p,
        env={
            "LIVEKIT_URL": "wss://from-env",
            "LIVEKIT_API_KEY": "env-key",
            "LIVEKIT_API_SECRET": "env-secret",
        },
    )
    # Secrets always come from the environment, even if the (untrusted,
    # possibly committed) YAML file has fields with the same names.
    assert cfg.url == "wss://from-env"
    assert cfg.api_key == "env-key"
    assert cfg.api_secret == "env-secret"


def test_from_yaml_non_mapping_top_level_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("- just\n- a\n- list\n")
    with pytest.raises(ConfigError):
        LiveKitConfig.from_yaml(p, env={})


def test_load_config_explicit_missing_path_is_hard_error():
    with pytest.raises(FileNotFoundError):
        load_config("nope/missing.yaml", env={})


def test_load_config_falls_back_to_env_only_when_no_default_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no configs/realtime_agent.yaml here
    cfg = load_config(env={"LIVEKIT_ROOM": "r2"})
    assert cfg.room_name == "r2"


def test_load_config_uses_default_yaml_when_present(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "realtime_agent.yaml").write_text("room_name: from-default-file\n")
    cfg = load_config(env={})
    assert cfg.room_name == "from-default-file"

from cli.commands.status import _check_parakeet_model


def test_parakeet_diagnostic_accepts_null_paths_config(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENVISION_HOME", str(tmp_path))

    result = _check_parakeet_model({"paths": None})

    assert result["path"] == str(tmp_path / "models" / "parakeet-tdt-0.6b-v3-int8")
    assert result["installed"] is False
    assert result["missing_files"] == ["model.onnx", "tokens.txt"]


def test_parakeet_diagnostic_accepts_null_models_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENVISION_HOME", str(tmp_path))

    result = _check_parakeet_model({"paths": {"models_dir": None}})

    assert result["path"] == str(tmp_path / "models" / "parakeet-tdt-0.6b-v3-int8")

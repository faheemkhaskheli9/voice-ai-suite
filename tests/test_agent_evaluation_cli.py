from pathlib import Path

from agent_evaluation.cli import main

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "personas"


def test_validate_directory_ok(capsys):
    assert main(["validate", str(EXAMPLES)]) == 0
    out = capsys.readouterr().out
    assert "frustrated_billing" in out
    assert "2 persona(s) valid" in out


def test_validate_single_file_ok():
    assert main(["validate", str(EXAMPLES / "curious_new_customer.yaml")]) == 0


def test_validate_bad_file_returns_1(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\n", encoding="utf-8")
    assert main(["validate", str(bad)]) == 1
    assert "error:" in capsys.readouterr().err


def test_show_outputs_json(capsys):
    assert main(["show", str(EXAMPLES / "frustrated_billing.yaml")]) == 0
    assert '"name": "frustrated_billing"' in capsys.readouterr().out

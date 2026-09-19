from pathlib import Path

import pytest

from agent_evaluation.persona import Persona, PersonaError, load_persona, load_personas

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples" / "personas"

VALID = """
name: tester
goal: Do the thing.
tone: calm
sample_utterances:
  - "Hello."
  - "Please help."
success_criteria:
  - "Agent helps."
"""


def _write(tmp_path, text, name="p.yaml"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_valid_persona_loads_with_defaults(tmp_path):
    persona = load_persona(_write(tmp_path, VALID))
    assert isinstance(persona, Persona)
    assert persona.name == "tester"
    assert persona.max_turns == 12
    assert persona.language == "en"
    assert persona.metadata == {}


def test_checked_in_examples_all_parse():
    personas = load_personas(EXAMPLES)
    assert {p.name for p in personas} == {"frustrated_billing", "curious_new_customer"}


def test_template_config_parses():
    load_persona(REPO_ROOT / "configs" / "persona_template.yaml")


def test_missing_required_field_raises(tmp_path):
    text = VALID.replace("tone: calm\n", "")
    with pytest.raises(PersonaError, match="tone"):
        load_persona(_write(tmp_path, text))


def test_empty_success_criteria_rejected(tmp_path):
    text = VALID.replace('  - "Agent helps."\n', "")
    with pytest.raises(PersonaError):
        load_persona(_write(tmp_path, text))


def test_blank_utterance_rejected(tmp_path):
    text = VALID.replace('  - "Hello."\n', '  - "   "\n')
    with pytest.raises(PersonaError, match="non-empty"):
        load_persona(_write(tmp_path, text))


def test_unknown_field_rejected(tmp_path):
    with pytest.raises(PersonaError, match="voice|extra|permitted|allowed"):
        load_persona(_write(tmp_path, VALID + "voice: robotic\n"))


def test_bad_yaml_reports_clearly(tmp_path):
    with pytest.raises(PersonaError, match="not valid YAML"):
        load_persona(_write(tmp_path, "name: x\n  bad: : :\n"))


def test_top_level_list_rejected(tmp_path):
    with pytest.raises(PersonaError, match="mapping"):
        load_persona(_write(tmp_path, "- a\n- b\n"))


def test_missing_file_raises():
    with pytest.raises(PersonaError, match="not found"):
        load_persona("does/not/exist.yaml")


def test_max_turns_out_of_range_rejected(tmp_path):
    with pytest.raises(PersonaError):
        load_persona(_write(tmp_path, VALID + "max_turns: 0\n"))

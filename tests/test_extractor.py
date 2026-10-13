import pytest

from app.config import Settings
from app.extractor import coerce_spec, extract
from app.llm import LLMError, LLMReply, parse_model_output

BRIEF = "18k gold ring with a single diamond, polished"
GOOD = '{"product_type":"ring","metal":"gold","purity":"18k","stone_type":"diamond","stone_count":1,"dimension_mm":null,"finish":"polished"}'


def settings(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "primary")
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "backup")
    return Settings()


def scripted(replies, calls):
    """Fake LLM: each element is a string reply or an Exception to raise."""
    it = iter(replies)

    def call(provider, model, key, system, user, image, timeout):
        calls.append(model)
        r = next(it)
        if isinstance(r, Exception):
            raise r
        return LLMReply(r, 10, 5)
    return call


@pytest.mark.parametrize("text,how", [
    (GOOD, "clean"),
    ("```json\n" + GOOD + "\n```", "repaired"),
    ("Sure! Here you go: " + GOOD + " Hope that helps", "repaired"),
    (GOOD.replace('"polished"}', '"polished",}'), "repaired"),
    ("{'product_type': 'ring', 'metal': 'gold'}", "repaired"),
    ("", "failed"), ("I cannot help with that", "failed"), ('{"product_type": "ring"', "failed"), ("[1,2,3]", "failed"),
])
def test_parse_model_output_never_raises(text, how):
    obj, got = parse_model_output(text)
    assert got == how
    assert (obj is None) == (how == "failed")


def test_malformed_output_is_repaired_and_reported(monkeypatch):
    calls = []
    fields, meta = extract(BRIEF, None, settings(monkeypatch), scripted(["```json\n" + GOOD + "\n```"], calls))
    assert meta.mode == "llm_repaired" and fields["product_type"].value == "ring"
    assert fields["product_type"].confidence == "medium"      # high is capped when the reply needed repair


def test_unparseable_then_good_reply_is_a_retry(monkeypatch):
    calls = []
    fields, meta = extract(BRIEF, None, settings(monkeypatch), scripted(["garbage", GOOD], calls))
    assert meta.mode == "llm_retry" and calls == ["primary", "primary"]


def test_persistent_garbage_degrades_to_rules_without_crashing(monkeypatch):
    calls = []
    fields, meta = extract(BRIEF, None, settings(monkeypatch), scripted(["nope"] * 4, calls))
    assert meta.mode == "rules_fallback"
    assert calls == ["primary", "primary", "backup", "backup"]
    assert fields["metal"].value == "gold" and fields["stone_count"].value == 1


def test_provider_error_uses_fallback_model(monkeypatch):
    calls = []
    fields, meta = extract(BRIEF, None, settings(monkeypatch), scripted([LLMError("503"), GOOD], calls))
    assert calls == ["primary", "backup"] and meta.model == "backup"
    assert any("fallback model" in n for n in meta.notes)


def test_empty_model_output_and_empty_brief(monkeypatch):
    fields, meta = extract("", None, settings(monkeypatch), scripted([], []))
    assert meta.mode == "empty_default" and all(f.value is None and f.confidence == "low" for f in fields.values())
    fields, meta = extract(BRIEF, None, settings(monkeypatch), scripted(["", ""] * 2, []))
    assert meta.mode == "rules_fallback"


def test_invalid_field_values_are_dropped_not_fatal():
    spec, issues = coerce_spec({"product_type": "spaceship", "metal": "Yellow Gold", "stone_count": "12", "dimension_mm": "-5", "finish": "unknown"})
    assert spec.product_type is None and spec.metal == "gold" and spec.stone_count == 12 and spec.dimension_mm is None
    assert len(issues) == 2


def test_llm_value_contradicting_the_brief_gets_low_confidence(monkeypatch):
    bad = GOOD.replace('"gold"', '"silver"')
    fields, _ = extract(BRIEF, None, settings(monkeypatch), scripted([bad], []))
    assert fields["metal"].value == "silver" and fields["metal"].confidence == "low"


def test_no_llm_configured_uses_rules(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    fields, meta = extract(BRIEF, None, Settings())
    assert meta.mode == "rules_fallback" and fields["stone_type"].value == "diamond"

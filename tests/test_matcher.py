import os

from app.matcher import AUTO_MIN, load_inventory, match, score_item

INV = load_inventory(os.path.join(os.path.dirname(__file__), "..", "data", "inventory.csv"))
SOLITAIRE = {"product_type": "ring", "metal": "gold", "purity": "18k", "stone_type": "diamond", "stone_count": 1, "dimension_mm": 18, "finish": "polished"}


def test_exact_spec_auto_selects_the_matching_sku():
    r = match(SOLITAIRE, INV)
    assert r.auto_selected == "R-001"
    assert r.candidates[0].score >= AUTO_MIN


def test_deterministic_same_input_same_ranking():
    runs = [[(c.sku, c.score) for c in match(SOLITAIRE, INV).candidates] for _ in range(5)]
    assert all(r == runs[0] for r in runs)
    assert [c.sku for c in match(SOLITAIRE, list(reversed(INV))).candidates] == [c.sku for c in match(SOLITAIRE, INV).candidates]  # order-independent


def test_partial_credit_never_auto_selects_wrong_metal():
    q2 = {"product_type": "pendant", "metal": "rose_gold", "purity": "14k", "stone_type": "diamond", "stone_count": 10}
    r = match(q2, INV)
    assert r.candidates[0].sku == "P-008"       # yellow-gold pendant ranks high on partial credit...
    assert r.auto_selected is None               # ...but must not be auto-selected
    assert "exact match" in r.reason


def test_unspecified_attributes_are_not_penalised_but_low_coverage_blocks_auto_select():
    r = match({"metal": "gold"}, INV)
    assert r.auto_selected is None and r.coverage < 0.5


def test_empty_query_returns_no_candidates():
    r = match({}, INV)
    assert r.candidates == [] and r.auto_selected is None


def test_product_type_mismatch_is_penalised():
    pendant = next(i for i in INV if i["sku"] == "P-001")
    ring_q = dict(SOLITAIRE)
    s_mismatch, _, _ = score_item(ring_q, pendant)
    assert s_mismatch < 0.5

import os

import cv2
from fastapi.testclient import TestClient

from app.cv import estimate_count
from app.main import app
from eval.make_images import render

client = TestClient(app)


def jpg(img):
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_cv_counts_a_clean_cluster():
    est = estimate_count(jpg(render("cluster", 7, "easy", 42)))
    assert est.status == "ok" and abs(est.count - 7) <= 1


def test_cv_rejects_corrupt_image_without_crashing():
    est = estimate_count(b"definitely not an image")
    assert est.status == "error"


def test_cv_blank_image_is_low_confidence():
    import numpy as np
    est = estimate_count(jpg(np.full((300, 300, 3), 200, np.uint8)))
    assert est.confidence == "low"


def test_api_conflict_between_brief_and_image_is_flagged():
    img = jpg(render("row", 12, "easy", 5))
    r = client.post("/analyze", data={"brief": "silver bracelet with 40 cz stones"}, files={"image": ("a.jpg", img, "image/jpeg")})
    body = r.json()
    assert r.status_code == 200 and body["needs_review"]
    assert any("disagree" in f["message"] for f in body["flags"])
    assert body["spec"]["stone_count"]["confidence"] == "low"


def test_api_empty_brief_does_not_crash_and_is_flagged():
    r = client.post("/analyze", data={"brief": ""})
    assert r.status_code == 200 and r.json()["needs_review"] and r.json()["extraction"]["mode"] == "empty_default"


def test_api_implausible_dimension_flagged():
    r = client.post("/analyze", data={"brief": "18k gold ring, 45mm diameter"})
    assert any(f["field"] == "dimension_mm" for f in r.json()["flags"])


def test_every_field_has_confidence_and_reason():
    r = client.post("/analyze", data={"brief": "silver pendant 20mm"}).json()
    assert set(r["spec"]) == {"product_type", "metal", "purity", "stone_type", "stone_count", "dimension_mm", "finish"}
    assert all(v["confidence"] in ("high", "medium", "low") and v["reason"] for v in r["spec"].values())

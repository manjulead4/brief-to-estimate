"""Classical (OpenCV) estimate of the number of small, roughly circular stones in a photo.

Two independent detectors; their agreement drives the confidence:
  A. Hough circle transform on the blurred greyscale image.
  B. Canny edges -> closed contours -> circularity/size filter -> merge duplicate (inner/outer outline) hits.
Both then apply a size-consistency filter (stones in one piece are similar in size) to reject glare dots / band edges.
"""
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .schemas import ImageEstimate

MAX_SIDE = 640
MAX_PLAUSIBLE_COUNT = 80   # beyond this the classical pipeline is not trusted (touching/overlapping stones)


def decode_image(data: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def _size_filter(circles: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    """Keep circles whose radius is within [0.6, 1.5] x the median radius (stones in one piece are similar in size)."""
    if len(circles) < 3:
        return circles
    med = float(np.median([c[2] for c in circles]))
    return [c for c in circles if 0.6 * med <= c[2] <= 1.5 * med]


def _dedupe(circles: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    kept: List[Tuple[float, float, float]] = []
    for c in sorted(circles, key=lambda c: -c[2]):   # larger (outer outline) first
        if all((c[0] - k[0]) ** 2 + (c[1] - k[1]) ** 2 > (0.7 * max(c[2], k[2])) ** 2 for k in kept):
            kept.append(c)
    return kept


def hough_circles(gray: np.ndarray) -> List[Tuple[float, float, float]]:
    """Primary detector. Parameters were grid-searched on a DEV set of synthetic images (seeds disjoint from the
    eval set): minDist ~ one stone diameter, maxRadius ~ H/20. They encode an assumption that stones are
    5-25 px radius in a ~512px image -- see Limitations in README."""
    h = gray.shape[0]
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    found = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(8, h // 36),
                             param1=100, param2=16, minRadius=max(4, h // 100), maxRadius=max(12, h // 20))
    if found is None:
        return []
    return _size_filter(_dedupe([tuple(map(float, c)) for c in found[0]]))


def contour_circles(gray: np.ndarray) -> List[Tuple[float, float, float]]:
    """Enclosed-region method: stone interiors are regions fenced in by edges. Counting those regions
    (rather than edge contours) keeps touching stones separate because the edge between them splits the regions."""
    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
    edges = cv2.Canny(blur, 30, 90)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    inner = (edges == 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inner, connectivity=4)
    h = gray.shape[0]
    rmin, rmax = max(4, h // 90), max(12, h // 14)
    out = []
    for i in range(1, n):
        x, y, w, hh, area = stats[i]
        if not (np.pi * (0.6 * rmin) ** 2 <= area <= np.pi * rmax ** 2) or max(w, hh) > 2.2 * rmax:
            continue
        mask = (labels[y:y + hh, x:x + w] == i).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        a, per = cv2.contourArea(c), cv2.arcLength(c, True)
        if a < 15 or per == 0:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        circ, fill = 4 * np.pi * a / (per * per), a / (np.pi * r * r + 1e-6)
        if circ > 0.7 and fill > 0.7 and min(w, hh) / max(w, hh) > 0.7:
            out.append((float(cx + x), float(cy + y), float(r)))
    return _size_filter(_dedupe(out))


def quality(gray: np.ndarray) -> Dict[str, float]:
    return {"sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()), "contrast": float(gray.std())}


def estimate_count(data: bytes) -> ImageEstimate:
    img = decode_image(data)
    if img is None:
        return ImageEstimate(status="error", reason="image could not be decoded")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    a, b = len(hough_circles(gray)), len(contour_circles(gray))
    q = quality(gray)
    methods: Dict[str, Any] = {"hough": a, "contour": b, **{k: round(v, 1) for k, v in q.items()}}
    # Both detectors are conservative (they miss stones more than they invent them), so take the larger count;
    # their agreement (within 20%) is the confidence signal. Chosen on a dev set, see EVALUATION.md.
    count = max(a, b)
    tol = max(2, int(round(0.2 * count)))
    agree = abs(a - b) <= tol
    if count == 0:
        return ImageEstimate(status="ok", count=0, confidence="low", methods=methods,
                             reason="no stone-like circles found by either detector (may be stones the baseline cannot see)")
    if count > MAX_PLAUSIBLE_COUNT:
        return ImageEstimate(status="ok", count=count, confidence="low", methods=methods,
                             reason="count above %d: dense/overlapping regime where the classical baseline is unreliable" % MAX_PLAUSIBLE_COUNT)
    if q["sharpness"] < 8 or q["contrast"] < 12:
        return ImageEstimate(status="ok", count=count, confidence="low", methods=methods,
                             reason="image is very blurry or low-contrast (sharpness %.1f, contrast %.1f)" % (q["sharpness"], q["contrast"]))
    if agree:
        return ImageEstimate(status="ok", count=count, confidence="high" if count <= 40 else "medium", methods=methods,
                             reason="Hough (%d) and contour (%d) detectors agree within +-%d" % (a, b, tol))
    return ImageEstimate(status="ok", count=count, confidence="low", methods=methods,
                         reason="detectors disagree: Hough=%d vs contour=%d" % (a, b))

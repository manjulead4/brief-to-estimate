"""Synthetic jewelry-style images with an exactly known stone count.

Layouts: 'arc' (stones along a ring band), 'row' (tennis bracelet), 'halo' (centre stone + ring of stones), 'cluster'.
Difficulty: easy (clean) | medium (noise + blur + gradient) | hard (touching stones, strong gradient, glare spots, JPEG).
Synthetic data is a limitation: real photos have reflections/occlusion this does not fully model (see EVALUATION.md).
"""
import math
import cv2
import numpy as np

SIZE = 512
STONE_COLORS = {"white": (235, 235, 240), "red": (60, 40, 190), "green": (80, 150, 60), "blue": (170, 90, 40)}


def _positions(layout, n, r, rng):
    cx, cy = SIZE // 2, SIZE // 2
    pts = []
    if layout == "arc":      # stones sit ON the band, along the top of the ellipse, spacing ~2.4r
        dt = 2.4 * r / 190.0
        pts = [(cx + 215 * math.cos(-math.pi / 2 + (i - (n - 1) / 2) * dt), cy + 20 + 170 * math.sin(-math.pi / 2 + (i - (n - 1) / 2) * dt)) for i in range(n)]
    elif layout == "row":
        step = min(2.4 * r, (SIZE - 60) / max(n, 1))
        x0 = cx - step * (n - 1) / 2
        pts = [(x0 + i * step, cy + 10 * math.sin(i / 3.0)) for i in range(n)]
    elif layout == "halo":
        k = n - 1
        if k >= 1:
            R = max(2.4 * r * k / (2 * math.pi), 2.2 * r)
            pts = [(cx + R * math.cos(2 * math.pi * i / k), cy + R * math.sin(2 * math.pi * i / k)) for i in range(k)]
        pts.append((cx, cy))
    else:  # cluster: poisson-ish placement with min distance
        tries = 0
        while len(pts) < n and tries < 5000:
            tries += 1
            p = (cx + rng.uniform(-130, 130), cy + rng.uniform(-110, 110))
            if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > (2.3 * r) ** 2 for q in pts):
                pts.append(p)
    return pts[:n]


def render(layout, n, difficulty, seed):
    rng = np.random.RandomState(seed)
    r = int(rng.randint(11, 17)) if n <= 14 else (9 if n <= 26 else 7)
    img = np.full((SIZE, SIZE, 3), (205, 205, 208), np.uint8)
    # background texture / gradient
    gx = np.linspace(0, 1, SIZE)[None, :, None]
    grad = {"easy": 0, "medium": 25, "hard": 55}[difficulty]
    img = np.clip(img.astype(np.float32) + (gx - 0.5) * grad, 0, 255).astype(np.uint8)
    # gold metal body: thick ellipse / band behind the stones
    cx, cy = SIZE // 2, SIZE // 2
    gold = (70, 150, 205)
    if layout == "arc":
        cv2.ellipse(img, (cx, cy + 20), (215, 170), 0, 0, 360, gold, 38)
    elif layout == "row":
        cv2.rectangle(img, (20, cy - 24), (SIZE - 20, cy + 34), gold, -1)
    elif layout == "halo":
        cv2.circle(img, (cx, cy), int(max(2.4 * r * max(n - 1, 1) / (2 * math.pi), 2.2 * r) + r + 14), gold, -1)
    else:
        cv2.ellipse(img, (cx, cy), (165, 140), 0, 0, 360, gold, -1)
    pts = _positions(layout, n, r, rng)
    cols = list(STONE_COLORS.values())
    palette_mode = rng.choice(["white", "mixed"])
    for (x, y) in pts:
        col = STONE_COLORS["white"] if palette_mode == "white" else cols[rng.randint(len(cols))]
        c = (int(x), int(y))
        cv2.circle(img, c, r + 3, (30, 70, 100), -1)      # dark bezel/setting
        cv2.circle(img, c, r, col, -1)
        cv2.circle(img, (c[0] - r // 3, c[1] - r // 3), max(r // 4, 2), (255, 255, 255), -1)  # facet highlight
    if difficulty == "hard":
        for _ in range(4):  # specular glare spots that are NOT stones
            gx0, gy0 = int(rng.randint(40, SIZE - 40)), int(rng.randint(40, SIZE - 40))
            cv2.circle(img, (gx0, gy0), int(rng.randint(2, 4)), (255, 255, 255), -1)
    if difficulty in ("medium", "hard"):
        img = cv2.GaussianBlur(img, (0, 0), 1.2 if difficulty == "medium" else 1.8)
        noise = rng.normal(0, 6 if difficulty == "medium" else 11, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def save(img, path, difficulty):
    q = 92 if difficulty != "hard" else 55
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, q])

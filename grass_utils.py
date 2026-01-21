import cv2
import numpy as np


def detect_grass_mask(img_bgr):
    """
    Finds grass pixels using a combo of Excess Green index and HSV filtering.
    Returns a binary mask where 255 = grass.
    
    The idea: real grass tends to be green in both RGB space (high G relative
    to R and B) and HSV space (hue in the green range). Combining both filters
    helps reject stuff like green cars or painted walls.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(img_bgr)

    # Excess Green index: 2G - R - B
    # High values = more green than you'd expect from a neutral gray
    exg = 2.0 * g.astype(np.float32) - r.astype(np.float32) - b.astype(np.float32)
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, exg_mask = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # HSV filter for green-ish hues
    lower = np.array([20, 40, 40], dtype=np.uint8)
    upper = np.array([90, 255, 255], dtype=np.uint8)
    hsv_mask = cv2.inRange(hsv, lower, upper)

    # Both conditions must be true
    mask = cv2.bitwise_and(exg_mask, hsv_mask)

    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Throw out tiny blobs (probably not actual lawn)
    h, w = mask.shape
    min_area = 0.05 * h * w
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    filtered = np.zeros_like(mask)
    for i in range(1, n_labels):  # skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255

    return filtered
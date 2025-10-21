"""Вспомогательные функции рисования рамок и масок"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Iterable, Optional, Tuple


def draw_transparent_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.45,
) -> None:
    """
    Overlaying a color mask on image pixels, where mask==1.

    Args:
        frame: HxWx3 BGR
        mask: HxW (bool/0-1/0-255)
        color: BGR
        alpha: прозрачность [0..1]
    """
    h, w = frame.shape[:2]

    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8)
    if mask.max() == 1:
        mask = mask * 255
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    color_img = np.zeros_like(frame, dtype=np.uint8)
    color_img[:] = color
    blended = cv2.addWeighted(color_img, alpha, frame, 1 - alpha, 0)

    cv2.copyTo(blended, mask, frame)



def draw_bbox_with_label(
    frame: np.ndarray,
    box: Iterable[float],
    label: str,
    color: Tuple[int, int, int] = (60, 160, 255),
    thickness: int = 2,
) -> None:
    """Draw a bbox with label """
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    th = int(th * 1.2)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 4)), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 2, y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

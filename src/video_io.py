"""Утилиты видео ввода/вывода для чтения кадров и записи аннотированного видео.
"""
from __future__ import annotations

import cv2
from typing import Tuple


def open_video_reader(path: str) -> cv2.VideoCapture:
    """Open a video file for reading.

    args:
        path: Path to the input video file.

    returns:
        cap: cv2.VideoCapture object.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    return cap


def get_video_props(cap: cv2.VideoCapture) -> Tuple[int, int, float]:
    """Get width, height, and fps from VideoCapture.
    """
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 1e-3:
        fps = 25.0
    return width, height, fps


def open_video_writer(path: str, width: int, height: int, fps: float) -> cv2.VideoWriter:
    """Open a video writer (mp4v codec)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {path}")
    return writer

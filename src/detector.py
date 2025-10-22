"""YOLO-based person detector/segmentor"""
from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
from ultralytics import YOLO


class PersonDetector:
    """Thin wrapper вокруг YOLO для обнаружения людей.
    Поддерживает как обнаружение (рамки), так и сегментацию экземпляров (маски) 
    в зависимости от типа переданной модели YOLO. """

    def __init__(
        self,
        model_path: str = "yolo11s.pt",
        conf: float = 0.35,
        iou: float = 0.5,
        imgsz: int = 640,
        device: Optional[str] = None,
    ) -> None:
        """Initialize YOLO model.

        Args:
            model_path: Путь или имя весовых коэффициентов YOLO.
            conf: Пороговое значение достоверности.
            iou: Пороговое значение IoU для NMS.
            imgsz: Размер выходного изображения.
            device: "cuda" / "cpu" / "cuda:0".
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        # task = getattr(self.model.model, "task", None)
        # print(task)

    def predict(self, frame: np.ndarray):
        """Run inference on a single BGR frame.

        Returns:
            Ultralytics Results list.
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
            classes=[0],  # COCO id 0 = person
        )
        return results[0]

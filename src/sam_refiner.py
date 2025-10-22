"""Покадровое уточнение маски с помощью SAM

Install:
    curl -L -o weights/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import torch

try:
    from segment_anything import sam_model_registry, SamPredictor  
except Exception:
    sam_model_registry = None
    SamPredictor = None

class SAMRefiner:
    """Mask refinement with SAM"""
    def __init__(self, checkpoint_path: str, model_type: str = "vit_b", device: str = "cpu") -> None:
        if sam_model_registry is None or SamPredictor is None:
            raise ImportError(
                "segment-anything is not installed. Install:\n"
                "  curl -L -o weights/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            )
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device)
        self.predictor = SamPredictor(sam)
        self.device = device

    @torch.no_grad()
    def refine_box(self, frame_bgr: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
        """Return mask for bbox"""
        rgb = frame_bgr[:, :, ::-1].copy()
        self.predictor.set_image(rgb)
        box = box_xyxy.astype(np.float32)
        masks, scores, _ = self.predictor.predict(
            point_coords=None, point_labels=None, box=box, multimask_output=True
        )
        best = int(np.argmax(scores))
        return masks[best].astype(np.uint8)

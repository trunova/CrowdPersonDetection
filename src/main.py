"""Entry point: чтение видео, запуск детектора, вывод выходных данных, сохранение аннотированного видео.

Usage:
    python -m src.main --video crowd.mp4 --out out.mp4 --use_masks --model yolo11s-seg.pt --conf 0.35
"""
from __future__ import annotations

import argparse
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path

from .video_io import open_video_reader, open_video_writer, get_video_props
from .detector import PersonDetector
from .visualize import draw_bbox_with_label, draw_transparent_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crowd Person Detection")
    parser.add_argument("--video", type=str, default="input/crowd.mp4", help="Path to input video (e.g., crowd.mp4)")
    parser.add_argument("--out", type=str, default="output/out.mp4", help="Path to output annotated video")
    parser.add_argument("--model", type=str, default="yolo11n-seg.pt", help="YOLO weights (e.g., yolo11s.pt or yolo11s-seg.pt)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size (longer side)")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cpu/cuda/cuda:0 (auto if None)")
    parser.add_argument("--use_masks", action="store_true", help="Draw instance masks (requires *-seg.pt model)")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame (for quick tests)")
    parser.add_argument("--sam_refine", default=False, action="store_true",
                        help="Use SAM to refine masks from detector boxes (no tracking)")
    parser.add_argument("--sam_checkpoint", type=str, default=None,
                        help="Path to sam_vit_*.pth checkpoint (required if --sam-refine)")
    parser.add_argument("--sam_model", type=str, default="vit_b",
                        choices=["vit_b", "vit_l", "vit_h"], help="SAM model type")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    detector = PersonDetector(
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        use_masks=args.use_masks,
        device=args.device,
    )

    cap = open_video_reader(args.video)
    width, height, fps = get_video_props(cap)
    writer = open_video_writer(args.out, width, height, fps)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    pbar = tqdm(total=total, desc="Processing", unit="fr")

    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if args.stride > 1 and (idx % args.stride != 0):
                # passthrough original frame if skipping
                writer.write(frame)
                idx += 1
                pbar.update(1)
                continue

            res = detector.predict(frame)

            sam_refiner = None
            if args.sam_refine:
                if not args.sam_checkpoint:
                    raise ValueError("--sam-refine requested, but --sam-checkpoint not provided")
                from .sam_refiner import SAMRefiner
                sam_refiner = SAMRefiner(
                    checkpoint_path=args.sam_checkpoint,
                    model_type=args.sam_model,
                    device=args.device or "cpu",
                )


            if sam_refiner is not None:
                # YOLO даёт боксы, SAM уточняет маски покадрово
                if res.boxes is not None:
                    for box, conf in zip(res.boxes.xyxy.cpu().numpy(),
                                         res.boxes.conf.cpu().numpy()):
                        m = sam_refiner.refine_box(frame, box)
                        label = f"person {conf:.2f}"
                        draw_transparent_mask(frame, m, color=(60, 160, 255), alpha=0.45)
                        draw_bbox_with_label(frame, box, label, color=(60, 160, 255))
            elif args.use_masks and hasattr(res, "masks") and res.masks is not None:
            # if args.use_masks and hasattr(res, "masks") and res.masks is not None:
                for box, cls, conf, m in zip(res.boxes.xyxy.cpu().numpy(),
                                             res.boxes.cls.cpu().numpy(),
                                             res.boxes.conf.cpu().numpy(),
                                             res.masks.data.cpu().numpy()):
                    label = f"person {conf:.2f}"
                    m_bin = (m > 0.65).astype(np.uint8)
                    draw_transparent_mask(frame, m_bin, color=(60, 160, 255), alpha=0.45)
                    draw_bbox_with_label(frame, box, label, color=(60, 160, 255))
            else:
                # Boxes only
                if res.boxes is not None:
                    for box, cls, conf in zip(res.boxes.xyxy.cpu().numpy(),
                                              res.boxes.cls.cpu().numpy(),
                                              res.boxes.conf.cpu().numpy()):
                        label = f"person {conf:.2f}"
                        draw_bbox_with_label(frame, box, label, color=(60, 160, 255))

            writer.write(frame)
            idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        writer.release()


if __name__ == "__main__":
    main()

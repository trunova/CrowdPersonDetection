# Crowd Person Detection (Video)

Детектор и рисование людей на видео (`crowd.mp4`) с использованием Python и Ultralytics YOLO.
Создаёт выходное видео с наложенными рамками/масками, именами классов и уровнями достоверности.


## Quickstart

```bash
# 1) Create & activate conda env (recommended)
conda env create -f environment.yml
conda activate crowddet

# 2) Run (YOLO)
python -m src.main --video input/crowd.mp4 --out output/out.mp4 --use-masks --model yolo11s-seg.pt 
# or boxes only:
python -m src.main --video input/crowd.mp4 --out output/out.mp4 --model yolo11s.pt 

```

> Веса `yolo11s(-seg).pt` подгружаются при первом запуске.
> cpu поддерживается, gpu будет использоваться автоматически, если доступен.
## Repo Layout
```
.
├── README.md
├── requirements.txt
└── src
    ├── main.py          # entry point
    ├── detector.py      # YOLO wrapper
    ├── visualize.py     # drawing utilities (boxes & masks)
    └── video_io.py      # video reader/writer helpers
```

## Notes
- Фильтр класса по умолчанию — только для людей (COCO id 0).
- FPS и разрешение выходных данных по умолчанию соответствуют входным.
- Для длинных видео можно использовать --stride N для выборки каждого N-го кадра для инференса.

## Report / Analysis
**Наблюдаемые проблемы**: 
- пропуски обнаружения в местах массового скопления людей
- человек, перекрытый другим объектом или человеком в какой то момент видео, перестаёт детектироваться на несколько кадров
- фрагментированные маски
- дрожание

---

## Improvement Plan

1. Сейчас для запуска на CPU используется `yolo11n(-seg)` для скорости, однако можно попробовать `yolo11m(-seg)` или `yolo11l(-seg)` для улучшения качества.


2. Для борьбы с отсутствием сегментационных масок на перекрытых другими объектами людях можно добавить трекер (например, ByteTrack / OC‑SORT), это также может быть полезно для стабилизации IDs.

6. **Fine-tuning**
- Небольшая тонкая настройка нескольких аннотированных кадров из данного видео.
- Дообучение модели на специализированных датасетах, с большим количеством кадров массового скопления людей, условиями видимости, освещённостью аналогичными условиям на целевом видео.




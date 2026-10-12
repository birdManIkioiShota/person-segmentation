"""SAM 2.1による人物の切り抜き処理。"""

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from PIL import Image, ImageFilter
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
    "sam2.1_hiera_large.pt"
)
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def load_predictor() -> SAM2ImagePredictor:
    if not torch.cuda.is_available():
        raise RuntimeError("PNGの処理にはCUDA対応GPUが必要です")

    checkpoint = Path.home() / ".cache/person-segmentation/sam2.1_hiera_large.pt"
    if not checkpoint.is_file():
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_suffix(".pt.part")
        try:
            print("SAM 2.1のモデル重みをダウンロードしています")
            urlretrieve(CHECKPOINT_URL, temporary)
            temporary.replace(checkpoint)
        finally:
            temporary.unlink(missing_ok=True)

    model = build_sam2(MODEL_CONFIG, str(checkpoint), device="cuda")
    return SAM2ImagePredictor(model)


def cutout_person(predictor: SAM2ImagePredictor, source: Path) -> Image.Image:
    """中心点から得た候補のうち、最大面積のマスクをAlphaに使用する。"""
    with Image.open(source) as original:
        image = original.convert("RGB")

    point = np.array([[image.width / 2, image.height / 2]], dtype=np.float32)
    with torch.inference_mode():
        predictor.set_image(image)
        masks, _, _ = predictor.predict(
            point_coords=point,
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=True,
        )

    if len(masks) == 0:
        raise ValueError("人物マスクが取得できませんでした")
    areas = np.count_nonzero(masks, axis=(1, 2))
    if areas.max() == 0:
        raise ValueError("人物領域が空です")
    mask = masks[int(np.argmax(areas))]
    alpha = Image.fromarray(mask.astype(np.uint8) * 255)
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=1))
    image.putalpha(alpha)
    return image

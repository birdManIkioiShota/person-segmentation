"""SAM 2.1で人物領域を背景透過PNGに変換し、それ以外のファイルをコピーする。"""

import argparse
import shutil
import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from PIL import Image, ImageFilter
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm


CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
    "sam2.1_hiera_large.pt"
)
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def load_predictor() -> SAM2ImagePredictor:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="SAM 2.1による人物の背景透過処理")
    parser.add_argument("--input", required=True, type=Path, help="入力ディレクトリ")
    parser.add_argument("--output", required=True, type=Path, help="出力ディレクトリ")
    args = parser.parse_args()

    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    if not input_dir.is_dir():
        parser.error(f"入力ディレクトリが存在しません: {input_dir}")
    if input_dir == output_dir or input_dir in output_dir.parents or output_dir in input_dir.parents:
        parser.error("入出力ディレクトリは重複しない別の場所を指定してください")

    files = sorted(path for path in input_dir.rglob("*") if path.is_file())
    png_files = [path for path in files if path.suffix.lower() == ".png"]
    if png_files and not torch.cuda.is_available():
        print("エラー: PNGの処理にはCUDA対応GPUが必要です", file=sys.stderr)
        return 1

    predictor = None
    if png_files:
        try:
            predictor = load_predictor()
        except Exception as exc:
            print(f"エラー: SAM 2.1の読み込みに失敗しました: {exc}", file=sys.stderr)
            return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    processed = copied = failed = 0
    for source in tqdm(files, desc="ファイル処理", unit="件"):
        relative = source.relative_to(input_dir)
        destination = output_dir / relative
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix.lower() != ".png":
                shutil.copy2(source, destination)
                copied += 1
                continue

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
            mask = masks[int(np.argmax(areas))]
            if areas.max() == 0:
                raise ValueError("人物領域が空です")
            alpha = Image.fromarray(mask.astype(np.uint8) * 255)
            alpha = alpha.filter(ImageFilter.GaussianBlur(radius=1))
            image.putalpha(alpha)
            image.save(destination)
            processed += 1
        except Exception as exc:
            failed += 1
            tqdm.write(f"失敗: {relative}: {exc}", file=sys.stderr)

    print(f"完了: PNG {processed}件 / その他コピー {copied}件 / 失敗 {failed}件")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

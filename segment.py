"""SAM 3で人物を切り抜き、その他のファイルをそのままコピーする。"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def main() -> int:
    parser = argparse.ArgumentParser(description="SAM 3による人物の背景透過処理")
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

    processor = None
    if png_files:
        try:
            model = build_sam3_image_model(device="cuda")
            processor = Sam3Processor(model, device="cuda")
        except Exception as exc:
            print(f"エラー: SAM 3の読み込みに失敗しました: {exc}", file=sys.stderr)
            return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    processed = copied = skipped = failed = 0
    for source in files:
        destination = output_dir / source.relative_to(input_dir)
        if destination.exists():
            skipped += 1
            print(f"スキップ（出力済み）: {source.relative_to(input_dir)}")
            continue

        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix.lower() != ".png":
                shutil.copy2(source, destination)
                copied += 1
                continue

            with Image.open(source) as original:
                image = original.convert("RGB")
            with torch.inference_mode():
                state = processor.set_image(image)
                result = processor.set_text_prompt(prompt="person", state=state)
            if result["scores"].numel() == 0:
                raise ValueError("人物マスクが検出されませんでした")

            index = int(result["scores"].argmax().item())
            # masks_logits はSAM 3ではシグモイド適用済みの連続値（0〜1）。
            alpha = result["masks_logits"][index, 0].detach().cpu().numpy()
            alpha = np.clip(alpha * 255, 0, 255).astype(np.uint8)
            image.putalpha(Image.fromarray(alpha))
            image.save(destination)
            processed += 1
        except Exception as exc:
            failed += 1
            print(f"失敗: {source.relative_to(input_dir)}: {exc}", file=sys.stderr)

    print(f"完了: PNG {processed}件 / その他コピー {copied}件 / スキップ {skipped}件 / 失敗 {failed}件")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

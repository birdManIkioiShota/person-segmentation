"""室内背景の自動取得と人物背景合成のプレビュー生成。"""

import argparse
import random
import sys
from pathlib import Path

from tqdm import tqdm

from processing.backgrounds import composite_person, ensure_backgrounds
from processing.segmentation import cutout_person, load_predictor


def main() -> int:
    parser = argparse.ArgumentParser(description="人物と室内背景を合成したプレビュー画像の生成")
    parser.add_argument("--input", required=True, type=Path, help="入力ディレクトリ")
    parser.add_argument("--output", required=True, type=Path, help="出力ディレクトリ")
    parser.add_argument("--backgrounds", required=True, type=Path, help="背景画像の保存先")
    parser.add_argument("--samples", type=int, default=10, help="入力画像1枚あたりの生成数（既定値: 10）")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samplesには1以上の整数を指定してください")

    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    background_dir = args.backgrounds.resolve()
    if not input_dir.is_dir():
        parser.error(f"入力ディレクトリが存在しません: {input_dir}")
    if input_dir == output_dir or input_dir in output_dir.parents or output_dir in input_dir.parents:
        parser.error("入力ディレクトリと出力ディレクトリは重複できません")
    if background_dir == output_dir or background_dir in output_dir.parents or output_dir in background_dir.parents:
        parser.error("背景画像ディレクトリと出力ディレクトリは重複できません")
    if background_dir == input_dir or background_dir in input_dir.parents or input_dir in background_dir.parents:
        parser.error("背景画像ディレクトリと入力ディレクトリは重複できません")

    png_files = sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".png")
    if not png_files:
        print("処理対象のPNG画像がありません")
        return 0

    try:
        print("背景画像: Poly Haven (CC0) https://polyhaven.com")
        backgrounds = ensure_backgrounds(background_dir)
        predictor = load_predictor()
    except Exception as exc:
        print(f"エラー: 背景またはモデルの準備に失敗しました: {exc}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random()
    saved = failed = 0
    digits = max(2, len(str(args.samples - 1)))
    for source in tqdm(png_files, desc="プレビュー生成", unit="枚"):
        relative = source.relative_to(input_dir)
        try:
            person = cutout_person(predictor, source)
            destination_dir = output_dir / relative.parent
            destination_dir.mkdir(parents=True, exist_ok=True)
            for index in range(args.samples):
                dest = destination_dir / f"{relative.stem}_{index:0{digits}d}.png"
                result = composite_person(person, rng.choice(backgrounds), rng)
                result.save(dest)
                saved += 1
        except Exception as exc:
            failed += 1
            tqdm.write(f"失敗: {relative}: {exc}", file=sys.stderr)

    print(f"完了: プレビュー {saved}枚 / 失敗 {failed}件")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

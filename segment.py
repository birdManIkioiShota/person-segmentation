"""人物切り抜きPNGの生成と付随ファイルのコピー。"""

import argparse
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

from processing.segmentation import cutout_person, load_predictor


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
            image = cutout_person(predictor, source)
            image.save(destination)
            processed += 1
        except Exception as exc:
            failed += 1
            tqdm.write(f"失敗: {relative}: {exc}", file=sys.stderr)

    print(f"完了: PNG {processed}件 / その他コピー {copied}件 / 失敗 {failed}件")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

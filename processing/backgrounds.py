"""Poly Havenの室内画像の準備と背景合成。"""

import hashlib
import json
import random
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image
from tqdm import tqdm

API_ROOT = "https://api.polyhaven.com"
USER_AGENT = "person-segmentation-background-preview/1.0"
SETTINGS_PATH = Path(__file__).resolve().parent.parent / "background_assets.json"


def _read_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=40) as response:
        return json.load(response)


def _download_tonemapped(asset_id: str, path: Path, max_width: int) -> None:
    file_info = _read_json(f"{API_ROOT}/files/{asset_id}")
    entry = file_info.get("tonemapped")
    if not entry or not entry.get("url") or not entry.get("md5"):
        raise ValueError(f"トーンマッピング済みJPGがありません: {asset_id}")

    raw_path = path.with_suffix(".download")
    temp_path = path.with_suffix(".tmp")
    try:
        request = Request(entry["url"], headers={"User-Agent": USER_AGENT})
        digest = hashlib.md5()
        with urlopen(request, timeout=90) as response, raw_path.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                digest.update(chunk)
        if digest.hexdigest().lower() != entry["md5"].lower():
            raise ValueError(f"ダウンロードデータのMD5が一致しません: {asset_id}")

        with Image.open(raw_path) as image:
            image.draft("RGB", (max_width, max_width // 2))
            image = image.convert("RGB")
            if image.width > max_width:
                height = max(1, round(image.height * max_width / image.width))
                image = image.resize((max_width, height), Image.Resampling.LANCZOS)
            image.save(temp_path, format="JPEG", quality=90)
        temp_path.replace(path)
    finally:
        raw_path.unlink(missing_ok=True)
        temp_path.unlink(missing_ok=True)


def ensure_backgrounds(directory: Path) -> list[Path]:
    """室内背景を用意する。取得対象IDは初回に保存して再利用する。"""
    with SETTINGS_PATH.open(encoding="utf-8") as stream:
        settings = json.load(stream)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "asset_ids.json"
    if manifest.is_file():
        asset_ids = json.loads(manifest.read_text(encoding="utf-8"))
    else:
        assets = _read_json(f"{API_ROOT}/assets?type=hdris&environment=indoor")
        asset_ids = sorted(
            asset_id for asset_id, info in assets.items()
            if info.get("type") == 0
            and info.get("attributes", {}).get("environment") == "indoor"
        )[: settings["limit"]]
        if not asset_ids:
            raise RuntimeError("室内HDRIが取得できませんでした")
        manifest.write_text(
            json.dumps(asset_ids, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    backgrounds = []
    for asset_id in tqdm(asset_ids, desc="背景画像準備", unit="枚"):
        path = directory / f"{asset_id}.jpg"
        if not path.is_file():
            _download_tonemapped(asset_id, path, settings["max_width"])
        backgrounds.append(path)
    return backgrounds


def composite_person(person: Image.Image, background_file: Path, rng: random.Random) -> Image.Image:
    """背景のランダム領域を切り抜き、人物を同じ位置へAlpha合成する。"""
    out_width, out_height = person.size
    with Image.open(background_file) as source:
        background = source.convert("RGB")

    width, height = background.size
    aspect = out_width / out_height
    max_crop_width = min(width, int(height * aspect))
    if max_crop_width < 1:
        raise ValueError(f"背景の画像サイズが不正です: {background_file}")
    scale = rng.uniform(0.55, 1.0)
    crop_width = max(1, round(max_crop_width * scale))
    crop_height = max(1, min(height, round(crop_width / aspect)))
    crop_width = min(width, max(1, round(crop_height * aspect)))

    left = rng.randint(0, width - crop_width)
    # 全天球画像の天頂・天底に近い領域を避ける。
    upper = max(0, min(height - crop_height, round(height * 0.25 - crop_height / 2)))
    lower = max(upper, min(height - crop_height, round(height * 0.75 - crop_height / 2)))
    top = rng.randint(upper, lower)
    background = background.crop((left, top, left + crop_width, top + crop_height))
    background = background.resize(person.size, Image.Resampling.LANCZOS)
    return Image.alpha_composite(background.convert("RGBA"), person).convert("RGB")

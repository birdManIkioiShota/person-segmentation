# 人物セグメンテーション

SAM 2.1を使用して人物を切り抜き、背景透過画像を生成するPythonプロジェクトです。室内背景をランダムに合成したプレビュー画像の生成にも対応しています。

## 動作環境

Python 3.12、uv、CUDA対応GPUが必要です。

## セットアップ

依存関係をインストールします。

```bash
SAM2_BUILD_CUDA=0 uv sync
```

SAM 2.1のモデルは初回実行時に自動ダウンロードされます。

## 実行方法

### 人物の切り抜き

```bash
uv run python segment.py --input ./input --output ./output
```

入力PNGから背景透過PNGを生成します。画像以外のファイルはディレクトリ構造を維持してコピーします。既存ファイルは上書きします。

### 背景合成プレビュー

```bash
uv run python preview_backgrounds.py \
    --input ./input \
    --backgrounds ./backgrounds \
    --output ./preview \
    --samples 10
```

入力PNGごとに、ランダムな背景を合成した画像を10枚生成します。背景画像が不足している場合は自動ダウンロードします。

出力画像は元のファイル名を維持し、拡張子の直前に連番を追加します。

```text
yaw_135_pitch_010.png
  → yaw_135_pitch_010_00.png
  → yaw_135_pitch_010_01.png
  ...
  → yaw_135_pitch_010_09.png
```

既存ファイルは上書きします。

## 背景データ

背景画像には[Poly Haven](https://polyhaven.com/)の室内HDRIを使用します。

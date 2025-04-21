# pyVisualizeMNIST

Inspired by https://github.com/okdalto/VisualizeMNIST.


# MNIST Visualizer

MNISTビジュアライザーは、手書き数字認識のためのニューラルネットワークの動作を視覚的に理解するためのツールです。ユーザーが描いた数字をリアルタイムで認識し、ネットワークの内部状態を可視化します。

## 機能

- 28x28ピクセルのキャンバスに数字を描画
- リアルタイムでのニューラルネットワーク推論
- ネットワークの各層（入力層、隠れ層、出力層）の視覚化
- 各数字（0〜9）の認識確率をバーグラフで表示
- デバッグ情報の表示

## 必要条件

- Python 3.6以上
- NumPy
- OpenCV (cv2)

## インストール方法

```bash
# 必要なライブラリをインストール
pip install numpy opencv-python
```

## 使用方法

1. リポジトリをクローンまたはダウンロードします
2. 重みとバイアスのファイルが正しいディレクトリにあることを確認します
3. スクリプトを実行します

```bash
python pyVisualizeMnist.py
```

## 操作方法

- 右側のキャンバスにマウスで数字を描きます
- 左側のパネルにニューラルネットワークの状態と予測結果が表示されます
- `c`キーでキャンバスをクリアします
- `ESC`キーでアプリケーションを終了します

## ファイル構成

- `mnist-visualizer-japanese.py` - メインのPythonスクリプト
- `data/weight1.txt` - 1層目の重み
- `data/weight2.txt` - 2層目の重み
- `data/weight3.txt` - 3層目の重み
- `data/biases1.txt` - 1層目のバイアス
- `data/biases2.txt` - 2層目のバイアス
- `data/biases3.txt` - 3層目のバイアス

## ニューラルネットワークの構造

このビジュアライザーは3層のニューラルネットワークを使用しています：

1. 入力層: 784ノード (28x28ピクセル画像)
2. 第1隠れ層: 128ノード
3. 第2隠れ層: 16ノード
4. 出力層: 10ノード (数字0〜9)

活性化関数としてReLUを使用し、出力層にはソフトマックス関数を適用しています。

## ONNX

![image](https://github.com/user-attachments/assets/a1ed7d8a-53da-42da-bb11-6d1999e745be)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class MNISTNetwork(nn.Module):
    def __init__(self):
        super(MNISTNetwork, self).__init__()
        
        # 元のネットワークと同じ構造を作成
        # 入力: 28x28=784 -> 第1隠れ層: 128 -> 第2隠れ層: 16 -> 出力: 10
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16, 10)
        
    def forward(self, x):
        # 入力を適切な形状に整形
        x = x.view(-1, 784)
        
        # レイヤー1: 線形変換 + ReLU
        x = F.relu(self.fc1(x))
        
        # レイヤー2: 線形変換 + ReLU
        x = F.relu(self.fc2(x))
        
        # レイヤー3: 線形変換
        x = self.fc3(x)
        
        # ソフトマックスは訓練時に損失関数に含まれるため、
        # 明示的にここで適用しない（ONNX出力用）
        # 必要に応じてソフトマックスを適用: F.softmax(x, dim=1)
        
        return x

def load_weight_from_txt(filename):
    """テキストファイルから重みを読み込む"""
    try:
        print(f"{filename}を読み込み中...")
        with open(filename, 'r') as f:
            content = f.read().strip()
        
        # 行で分割
        lines = content.split('\n')
        
        # データが1行の場合の処理（バイアス向け）
        if len(lines) == 1:
            values = lines[0].split(',')
            if values[-1] == '':  # 最後の要素がカンマの場合、空の文字列を削除
                values = values[:-1]
            
            # バイアスベクトルは一行で保存される
            return np.array([float(v) for v in values if v], dtype=np.float32)
        
        # 複数行の行列を処理（重み向け）
        rows = []
        for line in lines:
            if not line.strip():
                continue
            values = line.split(',')
            # 行が末尾にカンマで終わる場合、空の文字列を削除
            if values[-1] == '':
                values = values[:-1]
            rows.append([float(v) for v in values if v])
        
        return np.array(rows, dtype=np.float32)
            
    except Exception as e:
        print(f"{filename}の読み込みエラー: {e}")
        raise

def load_weights_and_create_model():
    """重みとバイアスを読み込み、モデルを作成する"""
    # モデルの初期化
    model = MNISTNetwork()
    
    # 重みとバイアスの読み込み
    w1 = load_weight_from_txt("weight1.txt")
    w2 = load_weight_from_txt("weight2.txt")
    w3 = load_weight_from_txt("weight3.txt")
    b1 = load_weight_from_txt("biases1.txt")
    b2 = load_weight_from_txt("biases2.txt")
    b3 = load_weight_from_txt("biases3.txt")
    
    print(f"読み込まれた重みの形状 - w1: {w1.shape}, w2: {w2.shape}, w3: {w3.shape}")
    print(f"読み込まれたバイアスの形状 - b1: {b1.shape}, b2: {b2.shape}, b3: {b3.shape}")
    
    # PyTorchの線形層は重みが(出力サイズ, 入力サイズ)の形状を想定する
    # NumPyとの互換性のために重みを転置する必要がある可能性がある
    # ここでは元のコードと一致するように重みとバイアスを設定
    
    # 重みとバイアスをPyTorch Tensorに変換
    # 必要に応じて転置を行う（元のコードがw * input の形式なら転置が必要）
    model.fc1.weight.data = torch.FloatTensor(w1.T)  # 転置が必要
    model.fc1.bias.data = torch.FloatTensor(b1)
    
    model.fc2.weight.data = torch.FloatTensor(w2.T)  # 転置が必要
    model.fc2.bias.data = torch.FloatTensor(b2)
    
    model.fc3.weight.data = torch.FloatTensor(w3.T)  # 転置が必要
    model.fc3.bias.data = torch.FloatTensor(b3)
    
    return model

def export_to_onnx(model, output_path="mnist_model.onnx"):
    """モデルをONNXフォーマットにエクスポート"""
    # ダミー入力テンソルを作成（バッチサイズ1、28x28ピクセル）
    dummy_input = torch.randn(1, 784)
    
    # モデルを評価モードに設定
    model.eval()
    
    # ONNX形式でエクスポート
    torch.onnx.export(
        model,               # エクスポートするモデル
        dummy_input,         # モデルの入力（例）
        output_path,         # 出力ファイルパス
        export_params=True,  # モデルのパラメータ（重みなど）をエクスポート
        opset_version=12,    # ONNXバージョン
        do_constant_folding=True,  # 定数畳み込みの最適化
        input_names=['input'],     # 入力の名前
        output_names=['output'],   # 出力の名前
        #dynamic_axes={
        #    'input': {0: 'batch_size'},   # バッチサイズを動的に
        #    'output': {0: 'batch_size'}
        #}
    )
    
    print(f"モデルを{output_path}にエクスポートしました")

if __name__ == "__main__":
    # モデルを作成し、重みを読み込む
    model = load_weights_and_create_model()
    
    # モデルをONNXにエクスポート
    export_to_onnx(model)
    
    # 動作確認のためのサンプル入力
    sample_input = torch.zeros(1, 784)
    # 中央に数字を描いたようなパターンを作成（サンプル用）
    sample_input[0, 784//2-30:784//2+30] = 1.0
    
    # 推論を実行
    with torch.no_grad():
        output = model(sample_input)
        probabilities = F.softmax(output, dim=1)
        predicted_digit = torch.argmax(probabilities, dim=1).item()
    
    print(f"サンプル入力に対する予測: {predicted_digit}")
    print(f"各クラスの確率: {probabilities.numpy()}")

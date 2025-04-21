import numpy as np
import cv2
import os

class MNISTVisualizer:
    def __init__(self):
        # 寸法の設定
        self.window_width = 1000
        self.window_height = 700
        self.canvas_width = 28
        self.canvas_height = 28
        
        # 描画用キャンバスの初期化
        self.canvas = np.zeros((self.canvas_height, self.canvas_width), dtype=np.uint8)
        
        # メインウィンドウの作成
        cv2.namedWindow('MNIST Visualizer')
        cv2.setMouseCallback('MNIST Visualizer', self.draw_on_canvas)
        
        # マウスの状態
        self.mouse_pressed = False
        self.prev_x = 0
        self.prev_y = 0
        
        # ニューラルネットワークの入出力
        self.input_mat = np.zeros((1, 784), dtype=np.float32)
        
        # デバッグ情報
        self.debug_info = []
        
        # 重みとバイアスの読み込み
        self.load_weights_and_biases()
        
    def load_weights_and_biases(self):
        """重みとバイアスをファイルから読み込む"""
        try:
            print("現在のディレクトリ:", os.getcwd())
            files = os.listdir()
            print("ディレクトリ内のファイル:", files)
            
            self.w1 = self.load_mat("data/weight1.txt")
            print("w1 shape:", self.w1.shape)
            
            self.w2 = self.load_mat("data/weight2.txt")
            print("w2 shape:", self.w2.shape)
            
            self.w3 = self.load_mat("data/weight3.txt")
            print("w3 shape:", self.w3.shape)
            
            self.b1 = self.load_mat("data/biases1.txt", transpose=True)
            print("b1 shape:", self.b1.shape)
            
            self.b2 = self.load_mat("data/biases2.txt", transpose=True)
            print("b2 shape:", self.b2.shape)
            
            self.b3 = self.load_mat("data/biases3.txt", transpose=True)
            print("b3 shape:", self.b3.shape)
            
        except Exception as e:
            print(f"モデルファイルの読み込みエラー: {e}")
            raise
        
    def load_mat(self, filename, transpose=False):
        """テキストファイルから行列を読み込む"""
        try:
            print(f"{filename}を読み込み中...")
            with open(filename, 'r') as f:
                content = f.read().strip()
            
            # 行で分割
            lines = content.split('\n')
            
            # データが1行の場合の処理
            if len(lines) == 1:
                values = lines[0].split(',')
                if values[-1] == '':  # 最後の要素がカンマの場合、空の文字列を削除
                    values = values[:-1]
                
                # バイアスベクトルは一般に単一行として保存される
                if transpose:
                    return np.array([float(v) for v in values if v], dtype=np.float32).reshape(1, -1)
                else:
                    # 必要に応じて列ベクトルとして返す
                    return np.array([float(v) for v in values if v], dtype=np.float32).reshape(-1, 1)
            
            # 複数行の行列を処理
            rows = []
            for line in lines:
                if not line.strip():
                    continue
                values = line.split(',')
                # 行が末尾にカンマで終わる場合、空の文字列を削除
                if values[-1] == '':
                    values = values[:-1]
                rows.append([float(v) for v in values if v])
            
            mat = np.array(rows, dtype=np.float32)
            if transpose:
                mat = mat.T
            
            return mat
            
        except Exception as e:
            print(f"{filename}の読み込みエラー: {e}")
            raise
        
    def draw_on_canvas(self, event, x, y, flags, param):
        """キャンバス上での描画用マウスイベントを処理"""
        # ウィンドウの座標からキャンバスの座標へのマッピング
        # ウィンドウの右側が描画エリア
        if x < self.window_width // 2:
            return  # 左側の可視化エリアでのクリックを無視
        
        # ウィンドウ座標をキャンバス座標に変換
        canvas_x = int((x - self.window_width // 2) * self.canvas_width / (self.window_width // 2))
        canvas_y = int(y * self.canvas_height / self.window_height)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_pressed = True
            self.prev_x = canvas_x
            self.prev_y = canvas_y
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_pressed = False
        elif event == cv2.EVENT_MOUSEMOVE and self.mouse_pressed:
            # キャンバス上に線を描画
            if 0 <= canvas_x < self.canvas_width and 0 <= canvas_y < self.canvas_height:
                cv2.line(self.canvas, (self.prev_x, self.prev_y), (canvas_x, canvas_y), 255, 2)
                self.prev_x = canvas_x
                self.prev_y = canvas_y
    
    def forward_pass(self):
        """ニューラルネットワークのフォワードパスを実行"""
        try:
            # キャンバスを入力形式に合わせてフラット化
            self.input_mat = self.canvas.flatten().reshape(1, -1).astype(np.float32) / 255.0
            
            # デバッグ情報
            self.debug_info = [
                f"shape: {self.input_mat.shape}",
                f"W1shape: {self.w1.shape}",
                f"b1shape: {self.b1.shape}"
            ]
            
            # レイヤー1
            mat1 = np.matmul(self.input_mat, self.w1)
            mat1 = mat1 + self.b1  # np.addの代わりに+演算子を使用
            mat1 = np.maximum(0, mat1)  # ReLU
            
            self.debug_info.append(f"layer1shape: {mat1.shape}")
            self.debug_info.append(f"W2shape: {self.w2.shape}")
            self.debug_info.append(f"b2shape: {self.b2.shape}")
            
            # レイヤー2
            mat2 = np.matmul(mat1, self.w2)
            mat2 = mat2 + self.b2
            mat2 = np.maximum(0, mat2)  # ReLU
            
            self.debug_info.append(f"layer2shape: {mat2.shape}")
            self.debug_info.append(f"W3shape: {self.w3.shape}")
            self.debug_info.append(f"b3shape: {self.b3.shape}")
            
            # レイヤー3
            mat3 = np.matmul(mat2, self.w3)
            mat3 = mat3 + self.b3
            
            self.debug_info.append(f"layer3shape: {mat3.shape}")
            
            # 出力確率のソフトマックス
            output = self.softmax(mat3)
            
            # 可視化のための整形
            reshaped_input = self.reshape_mat(self.input_mat, 28)
            reshaped_mat1 = self.reshape_mat(mat1, 8) if mat1.shape[1] == 64 else mat1.reshape(8, -1)
            reshaped_mat2 = self.reshape_mat(mat2, 4) if mat2.shape[1] == 16 else mat2.reshape(4, -1)
            
            return output, reshaped_input, reshaped_mat1, reshaped_mat2
            
        except Exception as e:
            print(f"フォワードパスでエラー: {e}")
            # クラッシュを防ぐためのプレースホルダーデータを返す
            return (np.zeros((1, 10)), 
                    np.zeros((28, 28)), 
                    np.zeros((8, 8)), 
                    np.zeros((4, 4)))
    
    def reshape_mat(self, mat, desired_col_num):
        """可視化のための1D行列を2D行列に整形"""
        try:
            if mat.shape[1] % desired_col_num != 0:
                print(f"整形エラー: {mat.shape[1]}は{desired_col_num}で割り切れません")
                # とにかく互換性のある形状を返す
                return mat.reshape(desired_col_num, -1)
            
            row = mat.shape[1] // desired_col_num
            result = np.zeros((desired_col_num, row), dtype=np.float32)
            
            idx = 0
            for i in range(desired_col_num):
                for j in range(row):
                    if idx < mat.shape[1]:
                        result[i, j] = mat[0, idx]
                        idx += 1
            
            return result
        except Exception as e:
            print(f"reshape_matでエラー: {e}")
            return np.zeros((desired_col_num, desired_col_num))
    
    def softmax(self, x):
        """ソフトマックス関数の計算"""
        try:
            e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return e_x / np.sum(e_x, axis=1, keepdims=True)
        except Exception as e:
            print(f"ソフトマックスでエラー: {e}")
            return np.ones((1, 10)) / 10  # フォールバックとして均一分布
    
    def create_2d_visualization(self, output, reshaped_input, reshaped_mat1, reshaped_mat2):
        """OpenCVを使用したニューラルネットワークの2D可視化を作成"""
        # 空の可視化画像を作成
        vis_height = self.window_height
        vis_width = self.window_width // 2
        visualization = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
        
        # 入力層を描画
        self.draw_layer_2d(visualization, reshaped_input, 50, 50, 10)
        
        # 最初の隠れ層を描画
        self.draw_layer_2d(visualization, reshaped_mat1, 50, 350, 15)
        
        # 2番目の隠れ層を描画
        self.draw_layer_2d(visualization, reshaped_mat2, 50, 500, 20)
        
        # 出力層と予測を描画
        self.draw_output_2d(visualization, output[0], 400, 100)
        
        # デバッグ情報を表示
        y_offset = 30
        for info in self.debug_info:
            cv2.putText(visualization, info, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            y_offset += 20
        
        return visualization
    
    def draw_layer_2d(self, img, mat, x_offset, y_offset, scale):
        """ニューラルネットワークの層を四角形のグリッドとして描画"""
        rows, cols = mat.shape
        cell_size = min(scale, 15)  # セルサイズを制限
        
        # セルのグリッドを描画
        for i in range(rows):
            for j in range(cols):
                # 位置を計算
                x = x_offset + j * cell_size
                y = y_offset + i * cell_size
                
                # 活性化に基づいて色を設定（白＝高活性化）
                intensity = int(mat[i, j] * 255)
                color = (intensity, intensity, intensity)
                
                # 塗りつぶされた四角形を描画
                cv2.rectangle(img, (x, y), (x + cell_size, y + cell_size), color, -1)
                
                # 境界線を描画
                cv2.rectangle(img, (x, y), (x + cell_size, y + cell_size), (50, 50, 50), 1)
    
    def draw_output_2d(self, img, output, x_offset, y_offset):
        """数字ラベルと予測バーを含む出力層を描画"""
        max_value = np.max(output)
        max_index = np.argmax(output)
        
        # 出力確率の棒グラフを描画
        for i in range(len(output)):
            # 位置を計算
            x = x_offset
            y = y_offset + i * 30
            
            # 数字ラベルを描画
            cv2.putText(img, f"{i}: {output[i]:.3f}", (x - 70, y + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # バーを描画（最大幅200pxに正規化）
            bar_width = int(output[i] * 200)
            
            # 最大値を強調表示
            color = (0, 200, 0) if i == max_index else (100, 100, 100)
            
            cv2.rectangle(img, (x, y), (x + bar_width, y + 20), color, -1)
            
        # 予測された数字を目立つように表示
        cv2.putText(img, f"prob: {max_index}", (x_offset, y_offset - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    
    def clear_canvas(self):
        """描画キャンバスをクリア"""
        self.canvas = np.zeros((self.canvas_height, self.canvas_width), dtype=np.uint8)
    
    def run(self):
        """メインアプリケーションループ"""
        while True:
            try:
                # ニューラルネットワークを処理
                output, reshaped_input, reshaped_mat1, reshaped_mat2 = self.forward_pass()
                
                # 2D可視化を作成
                visualization = self.create_2d_visualization(output, reshaped_input, reshaped_mat1, reshaped_mat2)
                
                # コンソールに予測を表示
                prediction = np.argmax(output[0])
                print(f"予測された数字: {prediction}、確信度: {output[0][prediction]:.3f}")
                
                # 表示用にキャンバスをリサイズ
                display_canvas = cv2.resize(self.canvas, (self.window_width // 2, self.window_height))
                
                # 表示用に単一チャンネルキャンバスを3チャンネルに変換
                display_canvas_3ch = cv2.cvtColor(display_canvas, cv2.COLOR_GRAY2BGR)
                
                # 最終表示ウィンドウを作成
                display = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
                display[:, :self.window_width//2] = visualization
                display[:, self.window_width//2:] = display_canvas_3ch
                
                # 表示を表示
                cv2.imshow('MNIST Visualizer', display)
                
                # キーボード入力を処理
                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # ESCキー
                    break
                elif key == ord('c'):
                    self.clear_canvas()
                    
            except Exception as e:
                print(f"メインループでエラー: {e}")
            
        cv2.destroyAllWindows()


if __name__ == "__main__":
    visualizer = MNISTVisualizer()
    visualizer.run()
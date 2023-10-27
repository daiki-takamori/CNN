import cv2
import numpy as np
import csv
import os

# CSVファイルのパス
csv_file = '/home/daiki/pytorch/box_marker/csv/output.csv'  # CSVファイルのパスを指定してください

# 画像が格納されているディレクトリのパス
image_directory = '/home/daiki/pytorch/box_marker/resize'  # 画像ファイルがあるディレクトリのパスを指定してください

# 出力先ディレクトリのパス
output_directory = '/home/daiki/pytorch/heatmap'  # 処理後の画像を保存するディレクトリのパスを指定してください

# ディレクトリが存在しない場合、作成する
os.makedirs(output_directory, exist_ok=True)

# CSVファイルを読み込む
with open(csv_file, mode='r', newline='') as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        # 画像ファイル名、x座標、y座標を取得
        image_filename = row['ファイル名']  # A列に画像ファイル名が入っていると仮定
        center_x = int(row['X'])  # B列にx座標が入っていると仮定
        center_y = int(row['Y'])  # C列にy座標が入っていると仮定
        
        # 画像を読み込む
        image_path = os.path.join(image_directory, image_filename)
        image = cv2.imread(image_path)
        
        if image is not None:
            # ヒートマップを生成
            heatmap_size = (image.shape[1], image.shape[0])
            heatmap = np.zeros(heatmap_size, dtype=np.float32)
            heatmap = cv2.GaussianBlur(heatmap, (0, 0), 100)
            heatmap[center_y, center_x] = 1.0
            
            # 色の設定（確率値が高いほど赤色、低いほど青色）
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            
            # ヒートマップを元の画像にオーバーレイ
            result = cv2.addWeighted(image, 0.7, heatmap_color, 0.3, 0)
            # 処理後の画像を保存
            output_path = os.path.join(output_directory, image_filename)
            cv2.imwrite(output_path, result)
            
            print(f"処理が完了しました：{output_path}")
        else:
            print(f"画像の読み込みに失敗しました：{image_path}")

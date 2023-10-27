from PIL import Image
import os
import cv2
import numpy as np
import csv

# 画像を格納しているディレクトリのパス
input_dir_nonclear = '/home/daiki/pytorch/box_marker/row/train'
input_dir_clear = '/home/daiki/pytorch/box_marker/removemarker/train'

# 画像を出力するディレクトリのパス
input_directory = '/home/daiki/pytorch/box_marker/resize/train'

# 重心点出力後のデータの出力先ディレクトリのパス
output_directory1 = '/home/daiki/pytorch/box_marker/marker_box/train'
output_directory2 = '/home/daiki/pytorch/box_marker/marker_cross/train'
output_directory3 = '/home/daiki/pytorch/box_marker/csv'

def nonclear():

    # 以下画像サイズ変更
    # 画像サイズを指定
    new_size = (224, 224)

    # ディレクトリ内のすべての画像ファイルに対して処理を実行
    for filename in os.listdir(input_dir_nonclear):
        if filename.endswith(".jpg"):  # または他のサポートされているファイル拡張子
            # 画像を開く
            img = Image.open(os.path.join(input_dir_nonclear, filename))

            # 画像サイズを変更
            img = img.resize(new_size)

            # 画像を保存
            img.save(os.path.join(input_directory, filename))

    # 以下画像の重心出力とデータセットcsvファイル作成
    # ディレクトリが存在しない場合、作成する
    os.makedirs(output_directory1, exist_ok=True)
    os.makedirs(output_directory2, exist_ok=True)

    # データを格納するためのリストを初期化
    data_list = []

    # CSVファイルを最初に1回だけ作成
    csv_filename = os.path.join(output_directory3, 'train.csv') #訓練データ or テストデータでファイル名変更

    # ディレクトリ内のすべてのファイルを取得
    for filename in os.listdir(input_directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # .jpg ファイルのみを処理
            
            # ファイルの絶対パスを取得
            filepath = os.path.join(input_directory, filename)
            
            # 画像を読み込み
            image = cv2.imread(os.path.join(input_directory, filename))

            # 画像をHSVカラースペースに変換
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # 赤色の範囲を定義
            lower_red = np.array([0, 100, 100])
            upper_red = np.array([10, 255, 255])

            # 赤色のマスクを作成
            mask = cv2.inRange(hsv, lower_red, upper_red)

            # 赤い領域を元の画像にマージ
            result_image = cv2.bitwise_and(image, image, mask=mask)

            # 赤色の物体を検出
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 面積でソート
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            # 最大の赤色領域を選択
            if len(contours) > 0:
                largest_contour = contours[0]

                # 重心座標を計算
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    # ファイル名、x座標、y座標の情報をリストに追加
                    data_list.append([filepath, cX, cY])

                    # ☓印を重心座標に描画
                    result_image_with_cross = result_image.copy()
                    marker_color = (255, 255, 255)  # ☓印の色 (BGR形式)
                    marker_size = 50  # ☓印のサイズ
                    cv2.drawMarker(result_image_with_cross, (cX, cY), marker_color, markerType=cv2.MARKER_CROSS, markerSize=marker_size, thickness=2)

                    # 赤色領域を四角で囲む
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # 画像を保存
                    cv2.imwrite(os.path.join(output_directory1, filename), result_image)
                    cv2.imwrite(os.path.join(output_directory2, 'cross_' + filename), result_image_with_cross)

                    # 完了メッセージ
                    print(f"ファイル {filename} の赤色領域処理が完了しました。")
            else:
                print(f"ファイル {filename} で赤色領域が見つかりませんでした。")

    # リスト内のデータをCSVファイルに書き込み
    with open(output_directory3 +'/train.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data_list)

def clear():
    # 画像ファイルの絶対パスを取得
    image_paths = [os.path.abspath(os.path.join(input_dir_clear, filename)) for filename in os.listdir(input_dir_clear) if filename.endswith(".jpg")]

    # CSVファイルに絶対パスを書き込む
    with open(output_directory3 +'/train.csv', "a", newline="") as file:
        csv_writer = csv.writer(file)
    
        # 画像ファイルの絶対パスをCSVファイルに書き込む
        for image_path in image_paths:
            csv_writer.writerow([image_path])

    print("CSVファイルに画像ファイルの絶対パスを記録しました。")

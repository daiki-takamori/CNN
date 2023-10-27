from PIL import Image
import os

# 画像を格納しているディレクトリのパス
input_dir = '/home/daiki/pytorch/box_marker/removemarker/test'

# 画像を出力するディレクトリのパス
output_dir = '/home/daiki/pytorch/box_marker/removemarker/test'

# 画像サイズを指定
new_size = (224, 224)

# ディレクトリ内のすべての画像ファイルに対して処理を実行
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # または他のサポートされているファイル拡張子
        # 画像を開く
        img = Image.open(os.path.join(input_dir, filename))

        # RGBA画像からRGBに変換
        
        img = img.convert('RGB')

        # 画像サイズを変更
        img = img.resize(new_size)

        # 画像を保存
        img.save(os.path.join(output_dir, filename))

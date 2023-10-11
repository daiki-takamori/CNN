# 必要ライブラリのインポート

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from PIL import Image
import os

# PyTorch関連ライブラリのインポート

import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torchviz import make_dot
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from tqdm.notebook import tqdm
from IPython.display import display

# warning表示off
import warnings
#warnings.simplefilter('ignore')

# デフォルトフォントサイズ変更
plt.rcParams['font.size'] = 14

# デフォルトグラフサイズ変更
plt.rcParams['figure.figsize'] = (6,6)

# デフォルトで方眼表示ON
plt.rcParams['axes.grid'] = True

# numpyの表示桁数設定
np.set_printoptions(suppress=True, precision=5)

# GPUチェック
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#以下諸関数定義

# 損失計算用
def eval_loss(loader, device, net, criterion):

    # データローダーから最初の1セットを取得する
    for images, labels in loader:
        break

    # デバイスの割り当て
    inputs = images.to(device)
    labels = labels.to(device)

    # 予測計算
    outputs = net(inputs)

    #  損失計算
    loss = criterion(outputs, labels)

    return loss

#正解データ数計算用
def calculate_distance(coord1, coord2):
    # 2つの座標間のユークリッド距離を計算する関数
  return ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2) ** 0.5

# 学習用関数
def fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history, max_error):

    # tqdmライブラリのインポート
    from tqdm.notebook import tqdm

    base_epochs = len(history)
    max_error_num=np.array([max_error, max_error])
    max_tensor=torch.tensor(max_error_num)
    max_tensor=max_tensor.to(device)

    for epoch in range(base_epochs, num_epochs+base_epochs):
        # 1エポックあたりの正解数(精度計算用)
        n_train_acc_pre, n_val_acc_pre = 0, 0
        n_train_acc, n_val_acc = 0, 0
        # 1エポックあたりの累積損失(平均化前)
        train_loss, val_loss = 0, 0
        # 1エポックあたりのデータ累積件数
        n_train, n_test = 0, 0

        #訓練フェーズ
        net.train()
        i=0

        for inputs, labels in tqdm(train_loader):
            # 1バッチあたりのデータ件数
            train_batch_size = len(labels)
            # 1エポックあたりのデータ累積件数
            n_train += train_batch_size

            # GPUヘ転送
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 勾配の初期化
            optimizer.zero_grad()

            # 予測計算
            outputs = net(inputs)

            # 損失計算
            loss = criterion(outputs, labels)

            # 勾配計算
            loss.backward()

            # パラメータ修正
            optimizer.step()

            # 予測ラベル導出
            #predicted = torch.max(outputs, 1)[1]
            
            # 予測座標を取得
            predicted_coords = outputs.detach().cpu().numpy()
            true_coords_np = labels.cpu().numpy()

            for i in range(len(true_coords_np)):
               
                # ユークリッド距離を計算
                error = calculate_distance(predicted_coords[i], true_coords_np[i])
                if error <= max_error:
                    n_train_acc_pre += 1  # 正解数をカウント
                
            # 平均前の損失と正解数の計算
            # lossは平均計算が行われているので平均前の損失に戻して加算
            train_loss += loss.item() * train_batch_size
            
            # 距離がmax_error以下であれば正解とみなす
            n_train_acc=n_train_acc_pre  
            


        #予測フェーズ
        net.eval()
        i=0

        for inputs_test, labels_test in test_loader:
            # 1バッチあたりのデータ件数
            test_batch_size = len(labels_test)
            # 1エポックあたりのデータ累積件数
            n_test += test_batch_size

            # GPUヘ転送
            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)

            # 予測計算
            outputs_test = net(inputs_test)

            # 損失計算
            loss_test = criterion(outputs_test, labels_test)

            # 予測ラベル導出
            #predicted_test = torch.max(outputs_test, 1)[1]

            # 予測座標を取得
            predicted_coords_val = outputs_test.detach().cpu().numpy()
            true_coords_np_val = labels_test.cpu().numpy()
            
            for i in range(len(true_coords_np_val)):
               
                # ユークリッド距離を計算
                error_val = calculate_distance(predicted_coords_val[i], true_coords_np_val[i])
                if error_val <= max_error:
                    n_val_acc_pre += 1  # 正解数をカウント
  
            #  平均前の損失と正解数の計算
            # lossは平均計算が行われているので平均前の損失に戻して加算
            val_loss +=  loss_test.item() * test_batch_size

            # 距離がmax_error以下であれば正解とみなす
            n_val_acc=n_val_acc_pre
        

        # 精度計算
        print(n_val_acc )
        print(n_test)
        train_acc = n_train_acc / n_train
        val_acc = n_val_acc / n_test
        # 損失計算
        avg_train_loss = train_loss / n_train
        avg_val_loss = val_loss / n_test
        # 結果表示
        print (f'Epoch [{(epoch+1)}/{num_epochs+base_epochs}], loss: {avg_train_loss:.5f} acc: {train_acc:.5f} val_loss: {avg_val_loss:.5f}, val_acc: {val_acc:.5f}')
        # 記録
        item = np.array([epoch+1, avg_train_loss, train_acc, avg_val_loss, val_acc])
        history = np.vstack((history, item))
    return history




# 学習ログ解析
def evaluate_history(history):
  #損失と精度の確認
  print(f'初期状態: 損失: {history[0,3]:.5f} 精度: {history[0,4]:.5f}') 
  print(f'最終状態: 損失: {history[-1,3]:.5f} 精度: {history[-1,4]:.5f}' )

  num_epochs = len(history)
  if num_epochs < 10:
    unit = 1
  else:
    unit = num_epochs / 10

  # 学習曲線の表示 (損失)
  plt.figure(figsize=(9,8))
  plt.plot(history[:,0], history[:,1], 'b', label='訓練')
  plt.plot(history[:,0], history[:,3], 'k', label='検証')
  plt.xticks(np.arange(0,num_epochs+1, unit))
  plt.xlabel('繰り返し回数')
  plt.ylabel('損失')
  plt.title('学習曲線(損失)')
  plt.legend()
  plt.show()

  # 学習曲線の表示 (精度)
  plt.figure(figsize=(9,8))
  plt.plot(history[:,0], history[:,2], 'b', label='訓練')
  plt.plot(history[:,0], history[:,4], 'k', label='検証')
  plt.xticks(np.arange(0,num_epochs+1,unit))
  plt.xlabel('繰り返し回数')
  plt.ylabel('精度')
  plt.title('学習曲線(精度)')
  plt.legend()
  plt.show()


# イメージとラベル表示
def show_images_labels(loader, classes, net, device):

    # DataLoaderから最初の1セットを取得する
    for images, labels in loader:
        break
    # 表示数は50個とバッチサイズのうち小さい方
    n_size = min(len(images), 50)

    if net is not None:
      # デバイスの割り当て
      inputs = images.to(device)
      labels = labels.to(device)

      # 予測計算
      outputs = net(inputs)
      predicted = torch.max(outputs,1)[1]
      #images = images.to('cpu')

    # 最初のn_size個の表示
    plt.figure(figsize=(20, 15))
    for i in range(n_size):
        ax = plt.subplot(5, 10, i + 1)
        label_name = classes[labels[i]]
        # netがNoneでない場合は、予測結果もタイトルに表示する
        if net is not None:
          predicted_name = classes[predicted[i]]
          # 正解かどうかで色分けをする
          if label_name == predicted_name:
            c = 'k'
          else:
            c = 'b'
          ax.set_title(label_name + ':' + predicted_name, c=c, fontsize=20)
        # netがNoneの場合は、正解ラベルのみ表示
        else:
          ax.set_title(label_name, fontsize=20)
        # TensorをNumPyに変換
        image_np = images[i].numpy().copy()
        # 軸の順番変更 (channel, row, column) -> (row, column, channel)
        img = np.transpose(image_np, (1, 2, 0))
        # 値の範囲を[-1, 1] -> [0, 1]に戻す
        img = (img + 1)/2
        # 結果表示
        plt.imshow(img)
        ax.set_axis_off()
    plt.show()


# PyTorch乱数固定用

def torch_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

#結果評価用(自作)
def calculate_accuracy(loader, model, device, max_error):
    model.eval()  # モデルを評価モードに設定
    正解数 = 0
    合計数 = 0

    with torch.no_grad():  # 評価中に勾配計算を無効化
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = outputs.cpu().numpy()  # 結果をCPUに移動し、NumPy配列に変換
            正解数 += np.sum(np.abs(predicted - labels.cpu().numpy()) <= max_error)  # 誤差がmax_error以内の場合に正解とみなす
            合計数 += labels.size(0)*2

    精度 = (正解数 / 合計数) * 100
    return 精度

def calculate_loss(loader, model, criterion, device):
    model.eval()  # モデルを評価モードに設定
    合計損失 = 0.0
    合計サンプル数 = 0

    with torch.no_grad():  # 評価中に勾配計算を無効化
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            損失 = criterion(outputs, labels)
            合計損失 += 損失.item() * labels.size(0)
            合計サンプル数 += labels.size(0)

    平均損失 = 合計損失 / 合計サンプル数
    return 平均損失

#結果表示用
def calculate_distance_orig(coord1, coord2):
    # 2つの座標間のユークリッド距離を計算する関数
    return ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2) ** 0.5

def show_predicted_results(loader, model, device, max_error):
    model.eval()  # モデルを評価モードに設定

    with torch.no_grad():  # 評価中に勾配計算を無効化
        for images, true_coords in loader:
            images = images.to(device)
            true_coords = true_coords.to(device)
            outputs = model(images)

            for i in range(len(images)):
                predicted_coords = outputs[i].cpu().numpy()  # 予測座標を取得
                true_coords_np = true_coords[i].cpu().numpy()  # 正解座標

                # 誤差を計算
                error = calculate_distance_orig(predicted_coords, true_coords_np)

                if error <= max_error:
                    prediction_label = "正解"
                else:
                    prediction_label = "不正解"

                print(f"正解座標: {true_coords_np}, 予測座標: {predicted_coords}, 予測結果: {prediction_label}, 誤差: {error:.2f}")

                # 画像を表示
                plt.figure(figsize=(5, 5))
                plt.imshow(np.transpose(images[i].cpu().numpy(), (1, 2, 0)))
                plt.axis('off')
                plt.show()



#以下CNN実装

#Transforms定義
custom_transform = transforms.Compose([
    transforms.Resize(112),
    transforms.CenterCrop(112),
    transforms.ToTensor(),
    #transforms.Normalize(0.5, 0.5)#-1~1に正規化する場合はこれ使う
])

#データセット作成
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.csv_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # 画像の幅と高さを取得（仮定：すべての画像が同じサイズ）
        self.image_width = 224  # 画像の幅に適した値に置き換えてください．ここはTransformにかける前の画素数
        self.image_height = 224  # 画像の高さに適した値に置き換えてください．ここはTransformにかける前の画素数

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.csv_data.iloc[idx, 0])  # CSVの1列目から画像ファイルのパスを取得
        image = Image.open(img_path)

        # xとy座標を0から1の範囲に正規化
        x = self.csv_data.iloc[idx, 1] / self.image_width
        y = self.csv_data.iloc[idx, 2] / self.image_height
        label = torch.tensor([x, y], dtype=torch.float32)  # 正規化したx, y座標のラベル

        if self.transform:
            image = self.transform(image)

        return image, label

#データセット作成
train_csv_file = '/home/daiki/pytorch/box_marker/csv/train.csv'  # CSVファイルのパス
train_root_dir = '/home/daiki/pytorch/box_marker/resize/train'    # 画像ファイルのルートディレクトリ
test_csv_file = '/home/daiki/pytorch/box_marker/csv/test.csv'  # CSVファイルのパス
test_root_dir = '/home/daiki/pytorch/box_marker/resize/test'    # 画像ファイルのルートディレクトリ

traindataset = CustomDataset(csv_file=train_csv_file, root_dir=train_root_dir, transform=custom_transform)
testdataset = CustomDataset(csv_file=test_csv_file, root_dir=test_root_dir, transform=custom_transform)

# データローダーを作成し、バッチごとにデータを取得します
batch_size = 5
train_dataloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(testdataset, batch_size=batch_size, shuffle=True)

'''
#データセット可視化
for batch in dataloader:
    images, coordinates = batch
    # ここでバッチデータを可視化する処理を行います

import matplotlib.pyplot as plt

# 最初のバッチを取得
batch = next(iter(dataloader))
images, coordinates = batch

# 画像を可視化
plt.figure(figsize=(10, 5))
for i in range(len(coordinates)):
    plt.subplot(2, len(coordinates)//2, i + 1)
    plt.imshow(images[i].permute(1, 2, 0))  # チャンネルの順序を変更して表示
    plt.title(f'Coordinates: {coordinates[i]}')
    plt.axis('off')


plt.tight_layout()
plt.show()

'''
#CNNモデル定義
# 入力次元数
#n_input = images_train.view(-1).shape[0]

# 出力次元数
n_output = 2

# 隠れ層のノード数
n_hidden = 128

# 結果確認
print(f'n_input: "一旦パス"  n_hidden: {n_hidden} n_output: {n_output}')

#CNNクラス作成
class CNN(nn.Module):
  def __init__(self, n_output, n_hidden):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 112, 3)
    self.conv2 = nn.Conv2d(112, 112, 3)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d((2,2))
    self.flatten = nn.Flatten()
    self.l1 = nn.Linear(326592, n_hidden)
    self.l2 = nn.Linear(n_hidden, n_output)

    self.features = nn.Sequential(
        self.conv1,
        self.relu,
        self.conv2,
        self.relu,
        self.maxpool)

    self.classifier = nn.Sequential(
       self.l1,
       self.relu,
       self.l2)

  def forward(self, x):
    x1 = self.features(x)
    x2 = self.flatten(x1)
    x3 = self.classifier(x2)
    return x3
  
#モデルインスタンス作成
# モデルインスタンス生成
net = CNN(n_output, n_hidden).to(device)

# 損失関数： 平均二乗誤差関数
criterion = nn.MSELoss()

# 学習率
lr = 0.01

# 最適化関数: 勾配降下法
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

# モデルの概要表示
print(net)

# モデルのサマリー表示
summary(net,(5,3,112,112),depth=1)

# 損失計算
loss = eval_loss(train_dataloader, device, net, criterion)

# 損失の計算グラフ可視化
g = make_dot(loss, params=dict(net.named_parameters()))
display(g)

#以下結果の評価

#学習
# 乱数初期化
torch_seed()

# モデルインスタンス生成
net = CNN(n_output, n_hidden).to(device)

# 損失関数： 平均二乗誤差関数
criterion = nn.MSELoss()

# 学習率
lr = 0.01

# 最適化関数: 勾配降下法
optimizer = optim.SGD(net.parameters(), lr=lr)

# 繰り返し回数
num_epochs = 100

# 評価結果記録用
history2 = np.zeros((0,5))

#誤差計算用の許容誤差範囲
max_error=0.1

# 学習
history2 = fit(net, optimizer, criterion, num_epochs, train_dataloader, test_dataloader, device, history2, max_error)

#評価
evaluate_history(history2)

# トレーニングループの後に、テストデータセットでモデルを評価します#現状いらない
#テスト精度 = calculate_accuracy(test_dataloader, net, device, max_error)
#テスト損失 = calculate_loss(test_dataloader, net, criterion, device)

#print(f"テスト精度: {テスト精度:.2f}%")
#print(f"テスト損失: {テスト損失:.4f}")

# 予測結果を表示する
show_predicted_results(test_dataloader, net, device, max_error)
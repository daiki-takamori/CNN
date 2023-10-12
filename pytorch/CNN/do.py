import make_dataset_train
import make_dataset_test

make_dataset_train.nonclear()
make_dataset_train.clear()
#make_dataset_test.nonclear()
#make_dataset_test.clear()

'''
import csv

# CSVファイルのパス
traincsv_file_path = '/home/daiki/pytorch/box_marker/csv/train.csv'
testcsv_file_path = '/home/daiki/pytorch/box_marker/csv/test.csv'

# 変数の初期化
traininputs = []
trainlabels = []
testinputs = []
testlabels = []

# CSVファイルを読み込む
with open(traincsv_file_path, 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        # 1列目の値をinputsに追加
        traininputs.append(row[0])
        
        # 2列目と3列目の値をlabelsに追加
        trainlabels.append([int(row[1]), int(row[2])])

# CSVファイルを読み込む
with open(testcsv_file_path, 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        # 1列目の値をinputsに追加
        testinputs.append(row[0])
        
        # 2列目と3列目の値をlabelsに追加
        testlabels.append([int(row[1]), int(row[2])])


# 結果を確認
print("traininputs:", traininputs)
print("trainlabels:", trainlabels)
print("testinputs:", testinputs)
print("testlabels:", testlabels)
print(len(trainlabels))
print(len(testlabels))

'''
from sklearn import datasets


# 1.取得手寫資料
mnist = datasets.load_digits()


x = mnist.images
x /= 255
y = mnist.target

# 顯示圖片
# import matplotlib.pyplot as plt
# plt.imshow(x[0], cmap='gray')
# plt.show()

# 2.建立 DataLoader
#     。將資料及分成 train、test
#     。將資料(Numpy)轉成Tensor
#     。將 (資料+標籤) 建成 Dataset
#     。建立 DataLoader
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

    # 2.1 將資料及分成 train、test
imgTrain, imgTest, tagTrain, tagTest = train_test_split(x, y, test_size=1/7, random_state=0)

    # 2.2 將資料(Numpy)轉成Tensor
imgTrain = torch.Tensor(imgTrain)
imgTest = torch.Tensor(imgTest)
tagTrain = torch.LongTensor(tagTrain)
tagTest = torch.LongTensor(tagTest)

    # 2.3 將 (資料+標籤) 建成 Dataset
dsTrain = TensorDataset(imgTrain, tagTrain)
dsTest = TensorDataset(imgTest, tagTest)

    # 2.4 建立 DataLoader
dlTrain = DataLoader(dsTrain,batch_size=64, shuffle=True)
dlTest = DataLoader(dsTest,batch_size=64, shuffle=False)

#3. 建置神經網路
from torch import nn
model = nn.Sequential(
    nn.Linear(8*8, 100) # 第一層 (輸入數量：8*8，輸出數量：100)
    , nn.ReLU() # 對前一層輸出，用 ReLU 函數處理
    , nn.Linear(100, 100)
    , nn.ReLU()
    , nn.Linear(100, 10)
)
print(model)
#4. 設定誤差函數、最佳化方法
from torch import optim
lossFn = nn.CrossEntropyLoss() # 分類問題，使用 「交叉熵誤差法」
optimizer = optim.Adam(model.parameters(), lr=0.01)

#5. train
model.train() # 將 model 轉訓練模式
epoch = 100
for e in range(epoch):
    for data, tag in dlTrain:
        data = data.view(-1, 64) # 將原本二維的資料(8X8)轉成一維(64)
        optimizer.zero_grad() # 每個批次訓練都先初始化
        output = model(data)
        loss = lossFn(output, tag)
        loss.backward()
        optimizer.step()
    print("epoch {0} : 結束".format(e))

#6. test
model.eval()
correct = 0
with torch.no_grad():
    for data, tag in dlTest:
        data = data.view(-1, 64)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        correct += predicted.eq(tag.data.view_as(predicted)).sum()

data_num = len(dlTest.dataset)
print('答對題數： {0}/{2} ; 正確率：{1}'.format(correct, 100 * correct / data_num, data_num))


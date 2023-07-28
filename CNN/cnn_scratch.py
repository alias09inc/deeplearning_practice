from time import time

import numpy as np
import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import torchsummary

use_cuda = torch.cuda.is_available()
print('Use CUDA:', use_cuda)

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5,), (0.5,))])

train_data = torchvision.datasets.MNIST(root="../data", train=True, transform=transform, download=True)
test_data = torchvision.datasets.MNIST(root="../data", train=False, transform=transform, download=True)

class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.l1 = nn.Linear(7*7*32, 1024)
        self.l2 = nn.Linear(1024, 1024)
        self.l3 = nn.Linear(1024, 10)
        self.lcustom = nn.Linear(7*7*32, 10)
        self.final = nn.Softmax()
    
    def forward(self, x):
        h = self.pool(self.activation(self.conv1(x)))
        h = self.pool(self.activation(self.conv2(h)))
        h = h.view(h.size()[0], -1)
        # h = self.lcustom(h)
        h = self.activation(self.l1(h))
        h = self.activation(self.l2(h))
        h = self.l3(h)
        # h = self.final(h)
        return h

model = CNN()
if use_cuda:
    model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.01)

if use_cuda:
    torchsummary.summary(model, (1, 28, 28), device='cuda')

# ミニバッチサイズ・エポック数の設定
batch_size = 100
epoch_num = 10

# データローダーの設定
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

# 誤差関数の設定 (必要に応じて誤差関数の計算もGPUで行うように設定変更)
criterion = nn.CrossEntropyLoss()
if use_cuda:
    criterion.cuda()

# ネットワークを学習モードへ変更
model.train()

# 学習の実行
train_start = time()
for epoch in range(1, epoch_num+1):   # epochのforループ
    # 1 epochの学習中の誤差・学習画像が正解した数をカウントする変数を初期化
    sum_loss = 0.0
    count = 0

    for image, label in train_loader:  # 1 epoch内のforループ (iterationのループ)

        if use_cuda:  # GPUで計算する場合は，データもGPUメモリ上へ移動させる
            image = image.cuda()
            label = label.cuda()

        y = model(image)  # データの入力と結果の出力

        # 誤差計算とbackpropagation, パラメータの更新
        loss = criterion(y, label)
        model.zero_grad()
        loss.backward()
        optimizer_adam.step()

        # 学習経過を確認するための処理
        sum_loss += loss.item()
        pred = torch.argmax(y, dim=1)
        count += torch.sum(pred == label)

    # 1 epoch終了時点での誤差の平均値，学習データに対する認識精度, 学習開始からの経過時間を表示
    print("epoch: {}, mean loss: {}, mean accuracy: {}, elapsed time: {}".format(epoch, sum_loss/len(train_loader), count.item()/len(train_data), time() - train_start))
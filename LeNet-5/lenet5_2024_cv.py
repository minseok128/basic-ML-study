# -*- coding: utf-8 -*-
"""LeNet5_2024_CV.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1czHChg6_znMExx2531xWKS0bh19_awuI
"""

import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from pytz import timezone

# 하이퍼파라미터 설정
RANDOM_SEED = 4242
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 15
IMG_SIZE = 32
NUM_CLASSES = 10


# 모델의 정확도를 계산하는 함수
def get_accuracy(model, data_loader, device):
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        model.eval()
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            _, probabilities = model(images)
            _, predicted_labels = torch.max(probabilities, 1)

            total_predictions += labels.size(0)
            correct_predictions += (predicted_labels == labels).sum()
    return correct_predictions.float() / total_predictions


# 학습 손실과 검증 손실을 시각화
def plot_loss(train_loss, val_loss):
    plt.style.use("grayscale")
    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    ax.plot(train_loss, color="green", label="Training Loss")
    ax.plot(val_loss, color="red", label="Validation Loss")
    ax.set(title="Loss Over Epochs", xlabel="EPOCH", ylabel="LOSS")
    ax.legend()
    fig.show()
    plt.style.use("default")


# 모델 학습 함수
def train(train_loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        logits, _ = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        loss.backward()
        optimizer.step()
    epoch_loss = total_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss


# 검증 데이터셋을 사용하여 모델의 성능을 평가
def validate(valid_loader, model, criterion, device):
    model.eval()
    total_loss = 0

    for images, labels in valid_loader:
        images = images.to(device)
        labels = labels.to(device)

        # 순전파와 손실 기록하기
        logits, _ = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)

    epoch_loss = total_loss / len(valid_loader.dataset)
    return model, epoch_loss


# 전체 학습 루프
def training_loop(
    model,
    criterion,
    optimizer,
    train_loader,
    valid_loader,
    epochs,
    device,
    print_every=1,
):

    best_loss = 1e10
    train_losses = []
    valid_losses = []

    for epoch in range(epochs):

        # training
        model, optimizer, train_loss = train(
            train_loader, model, criterion, optimizer, device
        )
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):

            train_acc = get_accuracy(model, train_loader, device=device)
            valid_acc = get_accuracy(model, valid_loader, device=device)

            print(
                datetime.now(timezone("Asia/Seoul")).time().replace(microsecond=0),
                "--- ",
                f"Epoch: {epoch}\t"
                f"Train loss: {train_loss:.4f}\t"
                f"Valid loss: {valid_loss:.4f}\t"
                f"Train accuracy: {100 * train_acc:.2f}\t"
                f"Valid accuracy: {100 * valid_acc:.2f}",
            )

    plot_loss(train_losses, valid_losses)

    return model, optimizer, (train_losses, valid_losses)


# LeNet5 모델 정의
class LeNet5(nn.Module):

    def __init__(self, num_classes):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=num_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probabilities = F.softmax(logits, dim=1)
        return logits, probabilities

# class LeNet5(nn.Module):
#     def __init__(self):
#         super(LeNet5, self).__init__()
#         self.c1 = nn.Conv2d(1, 6, kernel_size=5)
#         self.s2 = nn.AvgPool2d(2)

#         # C3 레이어는 16개의 출력을 생성하지만, 각 출력은 6개의 입력 중 일부 연결
#         self.c3 = nn.ModuleList([nn.Conv2d(3, 1, kernel_size=5) for _ in range(6)] + [nn.Conv2d(4, 1, kernel_size=5) for _ in range(6)] + [nn.Conv2d(6, 1, kernel_size=5) for _ in range(4)])
#         self.s4 = nn.AvgPool2d(2)
#         self.c5 = nn.Conv2d(16, 120, kernel_size=5)
#         self.fc1 = nn.Linear(120, 84)
#         self.fc2 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = F.tanh(self.c1(x))
#         x = self.s2(x)

#         # C3 레이어의 비대칭 연결
#         c3_outputs = []
#         c3_outputs.append(self.c3[0](x[:, [0, 1, 2], :, :]))
#         c3_outputs.append(self.c3[1](x[:, [1, 2, 3], :, :]))
#         c3_outputs.append(self.c3[2](x[:, [2, 3, 4], :, :]))
#         c3_outputs.append(self.c3[3](x[:, [3, 4, 5], :, :]))
#         c3_outputs.append(self.c3[4](x[:, [0, 4, 5], :, :]))
#         c3_outputs.append(self.c3[5](x[:, [0, 1, 5], :, :]))
#         c3_outputs.append(self.c3[6](x[:, [0, 1, 2, 3], :, :]))
#         c3_outputs.append(self.c3[7](x[:, [1, 2, 3, 4], :, :]))
#         c3_outputs.append(self.c3[8](x[:, [2, 3, 4, 5], :, :]))
#         c3_outputs.append(self.c3[9](x[:, [0, 3, 4, 5], :, :]))
#         c3_outputs.append(self.c3[10](x[:, [0, 1, 4, 5], :, :]))
#         c3_outputs.append(self.c3[11](x[:, [0, 1, 2, 5], :, :]))
#         c3_outputs.append(self.c3[12](x[:, [0, 1, 3, 4], :, :]))
#         c3_outputs.append(self.c3[13](x[:, [1, 2, 4, 5], :, :]))
#         c3_outputs.append(self.c3[14](x[:, [0, 2, 3, 5], :, :]))
#         c3_outputs.append(self.c3[15](x))

#         x = torch.cat(c3_outputs, dim=1)
#         x = self.s4(x)
#         x = F.tanh(self.c5(x))
#         x = torch.flatten(x, 1)
#         x = F.tanh(self.fc1(x))
#         x = self.fc2(x)
#         return x



# transforms 정의하기
transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

# 데이터셋 다운로드 및 생성
train_dataset = datasets.MNIST(
    root="mnist_data", train=True, transform=transform, download=True
)

valid_dataset = datasets.MNIST(root="mnist_data", train=False, transform=transform)

# 데이터 로더 정의
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 불러온 MNIST 데이터 확인
ROW_IMG = 10
N_ROWS = 5

fig = plt.figure()
for index in range(1, ROW_IMG * N_ROWS + 1):
    plt.subplot(N_ROWS, ROW_IMG, index)
    plt.axis("off")
    plt.imshow(train_dataset.data[index], cmap="gray_r")
fig.suptitle("MNIST Dataset - preview")

# 데이터셋 크기 출력
print(f"Train dataset size: {len(train_dataset)}")  # Train dataset size: 60000
print(
    f"Validation dataset size: {len(valid_dataset)}"
)  # Validation dataset size: 10000

torch.manual_seed(RANDOM_SEED)

model = LeNet5(NUM_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

model, optimizer, _ = training_loop(
    model, criterion, optimizer, train_loader, valid_loader, EPOCHS, DEVICE
)

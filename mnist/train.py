"""
MNIST 손글씨 분류 - 첫 딥러닝 학습 프로그램

구조:
  1. 데이터 준비   : MNIST 다운로드, 텐서 변환
  2. 모델 정의     : 간단한 완전연결 신경망(MLP)
  3. 학습 루프     : 순전파 -> 손실 계산 -> 역전파 -> 가중치 갱신
  4. 평가          : 테스트 데이터로 정확도 측정
  5. 저장          : 학습된 모델을 파일로 저장
"""
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------- 설정 ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5            # 전체 데이터를 몇 번 반복해서 볼지
BATCH_SIZE = 128      # 한 번에 처리하는 이미지 수
LR = 1e-3             # 학습률
DATA_DIR = "./data"
MODEL_PATH = "./mnist_mlp.pt"

print(f"장치: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU : {torch.cuda.get_device_name(0)}")


# ---------- 1. 데이터 ----------
# ToTensor: 0~255 픽셀을 0~1 실수로. Normalize: 평균 0.1307, 표준편차 0.3081로 정규화
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=1000)
print(f"학습 이미지 {len(train_ds)}장, 테스트 이미지 {len(test_ds)}장")


# ---------- 2. 모델 ----------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),           # 28x28 이미지를 784 벡터로 펼침
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),     # 0~9 숫자 10개 클래스
        )

    def forward(self, x):
        return self.net(x)


model = MLP().to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# ---------- 3. 학습 ----------
def train_one_epoch():
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()            # 이전 기울기 초기화
        outputs = model(images)          # 순전파
        loss = loss_fn(outputs, labels)  # 손실 계산
        loss.backward()                  # 역전파 (기울기 계산)
        optimizer.step()                 # 가중치 갱신
        total_loss += loss.item() * images.size(0)
    return total_loss / len(train_ds)


# ---------- 4. 평가 ----------
@torch.no_grad()
def evaluate():
    model.eval()
    correct = 0
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
    return correct / len(test_ds)


for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    loss = train_one_epoch()
    acc = evaluate()
    print(f"[epoch {epoch}/{EPOCHS}] loss={loss:.4f}  정확도={acc*100:.2f}%  ({time.time()-t0:.1f}s)")


# ---------- 5. 저장 ----------
torch.save(model.state_dict(), MODEL_PATH)
print(f"모델 저장: {MODEL_PATH}")

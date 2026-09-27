"""
학습된 MNIST 모델로 예측하는 프로그램 (실행기)
train.py 와 분리되어 있음 - 학습 없이 저장된 모델만 불러와서 사용
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./mnist_mlp.pt"


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


model = MLP().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
])
test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)

# 테스트 이미지 10장을 골라 예측 결과와 확신도(스코어) 출력
print("정답 | 예측 | 스코어")
with torch.no_grad():
    for i in range(10):
        image, label = test_ds[i]
        logits = model(image.unsqueeze(0).to(DEVICE))
        probs = torch.softmax(logits, dim=1)[0]
        pred = probs.argmax().item()
        mark = "" if pred == label else "  <- 틀림"
        print(f"  {label}  |  {pred}   | {probs[pred]*100:5.1f}%{mark}")

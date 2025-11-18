import os
import math
import json
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ARTIFACT_DIR = "./artifacts"
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3

# 손실 가중치(λ): Prototype Loss의 세기
LAMBDA_PL = 0.5 # 크면 클수록 임베딩을 대표점으로 더 끌어당김

# 프로토타입 EMA 계수(0~1): 클수록 과거값 유지, 작을수록 최근배치 반영
PROTO_EMA = 0.9 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset/Loader
class EmbeddingDataset(Dataset):
    """
    - 고정 임베딩(LLM에서 미리 추출)을 받아 학습
    - y: 0(neg), 1(pos)
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 모델 정의: 투영 + 소형 MLP 분류기
class ProjectionHead(nn.Module):
    """
    - 입력 임베딩(예: 1024차원) -> 투영 공간(예: 256차원)으로 변환
    - 목적: f(x) 공간에서 '응집/분리'를 더 잘 만들기
    """
    def __init__(self, in_dim: int, proj_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x):
        return self.net(x)


class ClassifierHead(nn.Module):
    """
    - f(x) -> 2 클래스 로짓 출력
    - 아주 단순한 1층 MLP로도 충분(투영이 이미 구분력을 만든다고 가정)
    """
    def __init__(self, proj_dim: int, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(proj_dim, num_classes)

    def forward(self, fx):
        return self.fc(fx)


# Prototype Loss
def prototype_loss(fx: torch.Tensor, y: torch.Tensor,
                   proto_neg: torch.Tensor, proto_pos: torch.Tensor) -> torch.Tensor:
    """
    L_pl = ||f(x) - m_c||^2 의 배치 평균
    - fx: (B, D) 투영 임베딩
    - y:  (B,)  레이블 0/1
    - proto_neg/pos: (D,) 각 클래스 프로토타입
    """
    # 타깃 프로토타입 선택
    target_proto = torch.where(
        (y == 0).unsqueeze(-1),
        proto_neg.unsqueeze(0),   # (1, D) -> (B, D)
        proto_pos.unsqueeze(0)
    )
    # L2 제곱 거리
    loss = ((fx - target_proto) ** 2).sum(dim=1).mean()
    return loss


# 프로토타입 EMA 갱신
@torch.no_grad()
def update_prototypes_ema(fx: torch.Tensor, y: torch.Tensor,
                          proto_neg: torch.Tensor, proto_pos: torch.Tensor,
                          ema: float = PROTO_EMA) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    - 현재 배치의 클래스별 평균으로 전역 프로토타입 갱신(EMA)
    - fx: (B, D), y: (B,)
    - proto_neg/pos: (D,)
    """
    if (y == 0).any():
        batch_mean_neg = fx[y == 0].mean(dim=0)
        proto_neg[:] = ema * proto_neg + (1 - ema) * batch_mean_neg
    if (y == 1).any():
        batch_mean_pos = fx[y == 1].mean(dim=0)
        proto_pos[:] = ema * proto_pos + (1 - ema) * batch_mean_pos
    return proto_neg, proto_pos


# 학습 루프
def train_loop(model_proj, model_clf, train_loader, val_loader,
               proto_neg, proto_pos, in_dim, proj_dim):
    ce_criterion = nn.CrossEntropyLoss()
    params = list(model_proj.parameters()) + list(model_clf.parameters())
    optimizer = torch.optim.Adam(params, lr=LR)

    best_val_acc = -1.0

    for epoch in range(1, EPOCHS + 1):
        model_proj.train(); model_clf.train()
        running_ce, running_pl, running_acc = 0.0, 0.0, 0.0
        n_samples = 0

        for X, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
            X = X.to(DEVICE)  # (B, in_dim)
            y = y.to(DEVICE)  # (B,)

            # 1) 투영
            fx = model_proj(X)  # (B, proj_dim)

            # 2) 분류 로짓
            logits = model_clf(fx)  # (B, 2)

            # 3) 손실 계산
            loss_ce = ce_criterion(logits, y)
            loss_pl = prototype_loss(fx, y, proto_neg, proto_pos)
            loss = loss_ce + LAMBDA_PL * loss_pl

            # 4) 역전파/최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 5) 프로토타입 EMA 갱신(스텝마다)
            with torch.no_grad():
                update_prototypes_ema(fx.detach(), y, proto_neg, proto_pos, PROTO_EMA)

            # 통계
            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean().item()

            bs = X.size(0)
            n_samples += bs
            running_ce += loss_ce.item() * bs
            running_pl += loss_pl.item() * bs
            running_acc += acc * bs

        # Epoch 통계
        epoch_ce = running_ce / n_samples
        epoch_pl = running_pl / n_samples
        epoch_acc = running_acc / n_samples

        # 검증
        val_acc = evaluate(model_proj, model_clf, val_loader)

        print(f"[Epoch {epoch}] "
              f"Train Acc={epoch_acc:.4f} | CE={epoch_ce:.4f} | PL={epoch_pl:.4f} || "
              f"Val Acc={val_acc:.4f}")

        # 모델 저장(최고 성능 갱신 시)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "proj": model_proj.state_dict(),
                "clf":  model_clf.state_dict(),
                "proto_neg": proto_neg.detach().cpu().numpy(),
                "proto_pos": proto_pos.detach().cpu().numpy(),
                "in_dim": in_dim,
                "proj_dim": proj_dim,
            }, os.path.join(ARTIFACT_DIR, "best_pn_classifier.pt"))
            print(f"  ↳ 모델 저장(Val 최고): {best_val_acc:.4f}")

    print(f"학습 종료. 최고 Val Acc={best_val_acc:.4f}")


@torch.no_grad()
def evaluate(model_proj, model_clf, loader):
    model_proj.eval(); model_clf.eval()
    total, correct = 0, 0
    for X, y in loader:
        X = X.to(DEVICE); y = y.to(DEVICE)
        fx = model_proj(X)
        logits = model_clf(fx)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += X.size(0)
    return correct / max(1, total)


def main():
    # 1) 임베딩 로드 (Train: PN만)
    X = np.load(os.path.join(ARTIFACT_DIR, "embeddings_train_X.npy"))
    y = np.load(os.path.join(ARTIFACT_DIR, "embeddings_train_y.npy"))  # 0/1

    # 2) Train/Val 분할 (PN 내부 분할)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    train_ds = EmbeddingDataset(X_train, y_train)
    val_ds   = EmbeddingDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    in_dim = X.shape[1]
    proj_dim = 256  # 투영 차원(하이퍼파라미터). 데이터에 맞게 128/256/384 등 실험 가능

    # 3) 모델 생성
    model_proj = ProjectionHead(in_dim=in_dim, proj_dim=proj_dim).to(DEVICE)
    model_clf  = ClassifierHead(proj_dim=proj_dim, num_classes=2).to(DEVICE)

    # 4) 프로토타입 초기화
    #    - 초기에는 Train 세트의 클래스 평균으로 설정(좋은 시작점)
    proto_neg_np = X_train[y_train == 0].mean(axis=0)  # (in_dim,)
    proto_pos_np = X_train[y_train == 1].mean(axis=0)

    #    - 투영 공간 프로토타입이므로, 초기 몇 배치 전까지는
    #      '입력 공간 평균'을 통과시킨 값으로 근사 초기화 or 0으로 시작 가능
    #    - 여기서는 안전하게 '0 벡터'로 두고 첫 몇 스텝동안 EMA로 빠르게 적응시킴
    proto_neg = torch.zeros(proj_dim, dtype=torch.float32, device=DEVICE, requires_grad=False)
    proto_pos = torch.zeros(proj_dim, dtype=torch.float32, device=DEVICE, requires_grad=False)

    # 5) 학습 루프(CE + λ*PL, 스텝마다 프로토타입 EMA 갱신)
    train_loop(model_proj, model_clf, train_loader, val_loader,
               proto_neg, proto_pos, in_dim, proj_dim)

    print("완료 ✅  (학습된 best_pn_classifier.pt 저장)")


if __name__ == "__main__":
    main()

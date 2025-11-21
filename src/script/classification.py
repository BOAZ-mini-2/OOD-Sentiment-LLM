import os
from pathlib import Path
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 설정
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3

# 손실 가중치(λ): Prototype Loss의 세기 (클수록 대표점으로 더 강하게 끌어당김)
LAMBDA_PL = 0.5

# 프로토타입 EMA 계수(0~1): 클수록 과거값 유지, 작을수록 최근 배치 반영↑
PROTO_EMA = 0.9

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_artifact_dir() -> Path:
    # 실행 위치와 무관하게 두 경로를 순회하여 자동 탐색
    candidates = [Path("src/artifacts"), Path("artifacts")]
    for cand in candidates:
        if (cand / "embeddings_train_X.npy").exists():
            return cand.resolve()
    raise FileNotFoundError(
        "embeddings_* 파일을 찾을 수 없습니다. 다음 폴더 중 하나에 두세요:\n"
        " - src/artifacts\n - artifacts\n"
        "필수 파일: embeddings_train_X.npy, embeddings_train_y.npy"
    )


# Dataset
class EmbeddingDataset(Dataset):
    """
    - 고정 임베딩(LLM에서 미리 추출한 벡터)을 받아 학습
    - y: 0(neg), 1(pos)
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 모델 정의: Projection + Classifier
class ProjectionHead(nn.Module):
    """
    - 입력 임베딩(예: 1536차원) -> 투영 공간(예: 256차원)으로 변환
    - 목적: f(x) 공간에서 '응집/분리'가 더 잘 일어나도록 비선형 변환
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
    - f(x) -> 2 클래스 로짓 출력 (간단한 선형 분류기)
    """
    def __init__(self, proj_dim: int, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(proj_dim, num_classes)

    def forward(self, fx):
        return self.fc(fx)


# Prototype Loss & EMA 갱신
def prototype_loss(
    fx: torch.Tensor,
    y: torch.Tensor,
    proto_neg: torch.Tensor,
    proto_pos: torch.Tensor
) -> torch.Tensor:
    """
    L_pl = || f(x) - m_c ||^2 의 배치 평균
    - fx:        (B, D) 투영 임베딩
    - y:         (B,)   레이블 0/1
    - proto_neg: (D,)   neg 프로토타입
    - proto_pos: (D,)   pos 프로토타입
    """
    target_proto = torch.where(
        (y == 0).unsqueeze(-1),     # (B,1)
        proto_neg.unsqueeze(0),     # (1,D) -> (B,D)
        proto_pos.unsqueeze(0)
    )
    loss = ((fx - target_proto) ** 2).sum(dim=1).mean()
    return loss


@torch.no_grad()
def update_prototypes_ema(
    fx: torch.Tensor,
    y: torch.Tensor,
    proto_neg: torch.Tensor,
    proto_pos: torch.Tensor,
    ema: float = PROTO_EMA
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    - 현재 배치의 클래스별 평균(fx)을 사용해 전역 프로토타입을 EMA로 갱신
    """
    if (y == 0).any():
        batch_mean_neg = fx[y == 0].mean(dim=0)
        proto_neg[:] = ema * proto_neg + (1 - ema) * batch_mean_neg
    if (y == 1).any():
        batch_mean_pos = fx[y == 1].mean(dim=0)
        proto_pos[:] = ema * proto_pos + (1 - ema) * batch_mean_pos
    return proto_neg, proto_pos


# 학습/평가 루프
def train_loop(
    model_proj: nn.Module,
    model_clf: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    proto_neg: torch.Tensor,
    proto_pos: torch.Tensor,
    artifact_dir: Path
):
    ce_criterion = nn.CrossEntropyLoss()
    params = list(model_proj.parameters()) + list(model_clf.parameters())
    optimizer = torch.optim.Adam(params, lr=LR)

    best_val_acc = -1.0

    for epoch in range(1, EPOCHS + 1):
        model_proj.train()
        model_clf.train()
        running_ce, running_pl, running_acc = 0.0, 0.0, 0.0
        n_samples = 0

        for X, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
            X = X.to(DEVICE)  # (B, in_dim)
            y = y.to(DEVICE)  # (B,)

            # 1) 투영
            fx = model_proj(X)  # (B, proj_dim)

            # 2) 분류 로짓
            logits = model_clf(fx)  # (B, 2)

            # 3) 손실 계산 (CE + λ * PL)
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

        print(
            f"[Epoch {epoch}] "
            f"Train Acc={epoch_acc:.4f} | CE={epoch_ce:.4f} | PL={epoch_pl:.4f} || "
            f"Val Acc={val_acc:.4f}"
        )

        # 모델 저장(최고 성능 갱신 시)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            artifact_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "proj": model_proj.state_dict(),
                    "clf":  model_clf.state_dict(),
                    "proto_neg": proto_neg.detach().cpu().numpy(),
                    "proto_pos": proto_pos.detach().cpu().numpy(),
                },
                artifact_dir / "best_pn_classifier.pt",
            )
            print(f"  ↳ 모델 저장(Val 최고): {best_val_acc:.4f}")

    print(f"학습 종료. 최고 Val Acc={best_val_acc:.4f}")


@torch.no_grad()
def evaluate(model_proj: nn.Module, model_clf: nn.Module, loader: DataLoader) -> float:
    model_proj.eval()
    model_clf.eval()
    total, correct = 0, 0
    for X, y in loader:
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        fx = model_proj(X)
        logits = model_clf(fx)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += X.size(0)
    return correct / max(1, total)


# =========================
# 메인
# =========================
def main():
    ART = find_artifact_dir()
    print("Using artifacts at:", ART)

    # 1) 임베딩 로드 (Train: PN만)
    X = np.load(ART / "embeddings_train_X.npy")  # (N, D)
    y = np.load(ART / "embeddings_train_y.npy")  # (N,)

    # 2) Train/Val 분할 (PN 내부 분할)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    train_ds = EmbeddingDataset(X_train, y_train)
    val_ds   = EmbeddingDataset(X_val, y_val)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False
    )

    in_dim = X.shape[1]
    proj_dim = 256  # 투영 차원(하이퍼파라미터: 128/256/384 등 실험 가능)

    # 3) 모델 생성
    model_proj = ProjectionHead(in_dim=in_dim, proj_dim=proj_dim).to(DEVICE)
    model_clf  = ClassifierHead(proj_dim=proj_dim, num_classes=2).to(DEVICE)

    # 4) 프로토타입 초기화
    #    - 투영 공간에서의 프로토타입을 바로 구하려면 투영 후 평균을 써야 하지만,
    #      여기서는 간단히 '0 벡터'로 시작하고 초기 스텝에서 EMA로 빠르게 수렴시킴.
    proto_neg = torch.zeros(proj_dim, dtype=torch.float32, device=DEVICE, requires_grad=False)
    proto_pos = torch.zeros(proj_dim, dtype=torch.float32, device=DEVICE, requires_grad=False)

    # 5) 학습 루프(CE + λ*PL, 스텝마다 프로토타입 EMA 갱신)
    train_loop(
        model_proj, model_clf,
        train_loader, val_loader,
        proto_neg, proto_pos,
        artifact_dir=ART
    )

    print("완료 ✅  (학습된 best_pn_classifier.pt 저장)")


if __name__ == "__main__":
    main()
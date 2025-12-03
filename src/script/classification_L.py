import os
from pathlib import Path
import numpy as np
from typing import Tuple, Union

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
LAMBDA_PL = 1.0

# 프로토타입 EMA 계수(0~1): 클수록 과거값 유지, 작을수록 최근 배치 반영↑
PROTO_EMA = 0.4

# 논문 식의 γ (온도/스케일 역할)
GAMMA = 1.0

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


# =========================
# Dataset
# =========================
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


# =========================
# 모델 정의: ProjectionHead
# =========================
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(

            # 1) Wide expansion
            nn.Linear(in_dim, 4096),
            nn.GELU(),
            nn.LayerNorm(4096),
            nn.Dropout(0.1),

            # 2) High-capacity layer
            nn.Linear(4096, 4096),
            nn.GELU(),
            nn.LayerNorm(4096),
            nn.Dropout(0.1),

            # 3) Begin funneling
            nn.Linear(4096, 2048),
            nn.GELU(),
            nn.LayerNorm(2048),

            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),

            nn.Linear(1024, 512),
            nn.GELU(),
            nn.LayerNorm(512),

            # 4) Final projection
            nn.Linear(512, proj_dim),   # 256
        )

    def forward(self, x):
        z = self.net(x)
        return z / (z.norm(p=2, dim=1, keepdim=True) + 1e-12)



# =========================
# Prototype-based logits
# =========================
def prototype_logits(
    fx: torch.Tensor,
    proto_neg: torch.Tensor,
    proto_pos: torch.Tensor,
    gamma: float = GAMMA
) -> torch.Tensor:
    """
    논문식:
      logit_c(x) = -γ * d(x, m_c)
      d(x, m_c) = ||f(x) - m_c||^2_2

    여기서는 C=2, K=1 (neg/pos 한 개씩) 특수 케이스.
    """
    # fx:        (B, D)
    # proto_neg: (D,)
    # proto_pos: (D,)

    diff_neg = fx - proto_neg.unsqueeze(0)   # (B, D)
    diff_pos = fx - proto_pos.unsqueeze(0)   # (B, D)

    dist_neg = (diff_neg ** 2).sum(dim=1)    # (B,)
    dist_pos = (diff_pos ** 2).sum(dim=1)    # (B,)

    # logits: (B, 2), [0]=neg, [1]=pos
    logits = torch.stack([-gamma * dist_neg, -gamma * dist_pos], dim=1)
    return logits


# =========================
# Prototype Loss & EMA 갱신
# =========================
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
def init_prototypes_from_data(
    model_proj: nn.Module,
    loader: DataLoader,
    proj_dim: int,
    device: str = DEVICE,
):
    """
    train_loader 전체를 한 번 훑어서
    클래스별 평균 f(x)를 초기 프로토타입으로 사용.
    """
    model_proj.eval()

    sum_neg = torch.zeros(proj_dim, device=device)
    sum_pos = torch.zeros(proj_dim, device=device)
    cnt_neg = 0
    cnt_pos = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        fx = model_proj(X)  # (B, proj_dim), L2 norm 포함

        mask_neg = (y == 0)
        mask_pos = (y == 1)

        if mask_neg.any():
            sum_neg += fx[mask_neg].sum(dim=0)
            cnt_neg += int(mask_neg.sum().item())

        if mask_pos.any():
            sum_pos += fx[mask_pos].sum(dim=0)
            cnt_pos += int(mask_pos.sum().item())

    # 평균
    proto_neg = sum_neg / max(cnt_neg, 1)
    proto_pos = sum_pos / max(cnt_pos, 1)

    # L2 정규화 (f(x)가 이미 정규화된 상태이므로 일관성 맞추기)
    proto_neg = proto_neg / (proto_neg.norm() + 1e-12)
    proto_pos = proto_pos / (proto_pos.norm() + 1e-12)

    return proto_neg, proto_pos



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


# =========================
# 학습/평가 루프
# =========================
def train_loop(
    model_proj: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    proto_neg: torch.Tensor,
    proto_pos: torch.Tensor,
    artifact_dir: Path
):
    ce_criterion = nn.CrossEntropyLoss()
    # 이제 projection head만 학습
    params = list(model_proj.parameters())
    optimizer = torch.optim.Adam(params, lr=LR)

    best_val_acc = -1.0

    for epoch in range(1, EPOCHS + 1):
        model_proj.train()
        running_ce, running_pl, running_acc = 0.0, 0.0, 0.0
        n_samples = 0

        for X, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
            X = X.to(DEVICE)  # (B, in_dim)
            y = y.to(DEVICE)  # (B,)

            # 1) 투영
            fx = model_proj(X)  # (B, proj_dim)  # 내부에서 L2 정규화까지 수행

            # 2) 분류 로짓 (prototype distance 기반)
            logits = prototype_logits(fx, proto_neg, proto_pos, gamma=GAMMA)

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
        val_acc = evaluate(model_proj, val_loader, proto_neg, proto_pos)

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
                    "proto_neg": proto_neg.detach().cpu().numpy(),
                    "proto_pos": proto_pos.detach().cpu().numpy(),
                },
                artifact_dir / "best_pn_classifier.pt",
            )
            print(f"  ↳ 모델 저장(Val 최고): {best_val_acc:.4f}")

    print(f"학습 종료. 최고 Val Acc={best_val_acc:.4f}")


@torch.no_grad()
def evaluate(
    model_proj: nn.Module,
    loader: DataLoader,
    proto_neg: torch.Tensor,
    proto_pos: torch.Tensor,
) -> float:
    model_proj.eval()
    total, correct = 0, 0
    for X, y in loader:
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        fx = model_proj(X)  # (B, proj_dim), L2 정규화 포함

        # 프로토타입 거리 기반 로짓
        logits = prototype_logits(fx, proto_neg, proto_pos, gamma=GAMMA)

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += X.size(0)
    return correct / max(1, total)


# =========================
# Mahalanobis용 mu / inv_cov export
# =========================
def export_md_params_from_trained_model(
    X: np.ndarray,
    ckpt_path: Union[str, Path],
    out_path: Union[str, Path] = "md_params.pt",
    proj_dim: int = 256,        # 학습할 때 사용한 proj_dim과 동일해야 함
    batch_size: int = 512,
    reg_eps: float = 1e-5,
) -> None:
    """
    학습이 끝난 small NN(ProjectionHead)에 train X를 다시 통과시켜
    Mahalanobis용 mu, inv_cov를 계산하고 파일로 저장하는 함수.
    (로컬 파이썬 기준: 다운로드 없음, 파일 저장만 수행)
    """
    ckpt_path = Path(ckpt_path)
    out_path = Path(out_path)

    if X.ndim != 2:
        raise ValueError(f"`X` must be 2D (N, in_dim), got shape {X.shape}")

    in_dim = X.shape[1]

    # 1) ProjectionHead 생성 후, 학습된 가중치 로드
    model_proj = ProjectionHead(in_dim=in_dim, proj_dim=proj_dim).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    if "proj" not in ckpt:
        raise KeyError(f"`proj` key not found in checkpoint: {ckpt_path}")

    model_proj.load_state_dict(ckpt["proj"])
    model_proj.eval()

    # 2) train X를 다시 통과시켜 f(x) 추출
    feats_list = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(DEVICE)
            fx = model_proj(xb)           # (B, proj_dim), L2 정규화 포함
            feats_list.append(fx.cpu().numpy())

    feats_train = np.vstack(feats_list)   # (N_train, proj_dim)

    # 3) mu, cov(+정규화), inv_cov 계산
    mu = feats_train.mean(axis=0)                     # (D,)
    cov = np.cov(feats_train, rowvar=False)          # (D, D)

    if reg_eps is not None and reg_eps > 0:
        cov = cov + reg_eps * np.eye(cov.shape[0], dtype=cov.dtype)

    inv_cov = np.linalg.pinv(cov)                    # (D, D)

    # 4) dict로 묶어서 .pt로 저장
    md_params = {
        "mu": mu,
        "inv_cov": inv_cov,
        "reg_eps": reg_eps,
        "proj_dim": proj_dim,
    }

    torch.save(md_params, out_path)

    print(f"[export_md_params] Saved mu & inv_cov to local file: {out_path}")
    print(f"  - mu shape      : {mu.shape}")
    print(f"  - inv_cov shape : {inv_cov.shape}")
    print(f"  - reg_eps       : {reg_eps}")


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

    # 4) 프로토타입 초기화: train_loader 전체 평균으로 설정
    proto_neg, proto_pos = init_prototypes_from_data(
        model_proj=model_proj,
        loader=train_loader,
        proj_dim=proj_dim,
        device=DEVICE,
    )

    # 5) 학습 루프(CE + λ*PL, 스텝마다 프로토타입 EMA 갱신)
    train_loop(
        model_proj,
        train_loader, val_loader,
        proto_neg, proto_pos,
        artifact_dir=ART
    )

    print("완료 ✅  (학습된 best_pn_classifier.pt 저장)")


if __name__ == "__main__":

    # 1) 학습
    main()

    # 2) Mahalanobis용 mu / inv_cov export
    ART = find_artifact_dir()

    X = np.load(ART / "embeddings_train_X.npy")  # (N, D)
    y = np.load(ART / "embeddings_train_y.npy")  # (N,)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    export_md_params_from_trained_model(
        X=X_train,
        ckpt_path=ART / "best_pn_classifier.pt",
        out_path=ART / "md_params.pt",
        proj_dim=256
    )

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

# 0) 아티팩트 경로 자동 탐색
CANDIDATES = [Path("src/artifacts"), Path("artifacts")]
ARTIFACT_DIR = None
for cand in CANDIDATES:
    if (cand / "embeddings_train_X.npy").exists():
        ARTIFACT_DIR = cand.resolve()
        break
if ARTIFACT_DIR is None:
    raise FileNotFoundError("embeddings_* 파일을 찾을 수 없습니다. 'src/artifacts' 또는 'artifacts'에 위치시켜 주세요.")

# 1) 임베딩 로드 (Train: PN만, OOD: 라벨=2)
train_X = np.load(ARTIFACT_DIR / "embeddings_train_X.npy")  # (N_pn, D)
train_y = np.load(ARTIFACT_DIR / "embeddings_train_y.npy")  # (N_pn,) 0=neg,1=pos
ood_X   = np.load(ARTIFACT_DIR / "embeddings_ood_X.npy")    # (N_ood, D)

# 2) 클래스별 프로토타입(평균 벡터)
proto_neg = train_X[train_y == 0].mean(axis=0, keepdims=True)  # (1, D)
proto_pos = train_X[train_y == 1].mean(axis=0, keepdims=True)  # (1, D)

# 3) L2 거리 → 각 샘플이 "가장 가까운 프로토타입"까지의 최소거리
def l2(a, b):
    return np.sqrt(((a - b) ** 2).sum(axis=-1))

dist_train_min = np.minimum(l2(train_X, proto_neg), l2(train_X, proto_pos))  # (N_pn,)
dist_ood_min   = np.minimum(l2(ood_X,   proto_neg), l2(ood_X,   proto_pos))  # (N_ood,)

# 4) 히스토그램 그리기
plt.figure(figsize=(9, 5))
plt.hist(dist_train_min, bins=50, alpha=0.6, label="IND (PN, min-dist→prototype)")
plt.hist(dist_ood_min,   bins=50, alpha=0.6, label="OOD (label=2, min-dist→prototype)")
plt.xlabel("Min distance to nearest PN prototype")
plt.ylabel("Count")
plt.title("PN vs OOD distance histogram (prototype-based)")
plt.legend()
plt.tight_layout()
png_path = ARTIFACT_DIR / "hist.png"
plt.savefig(png_path, dpi=150)
print(f"히스토그램 저장: {png_path}")
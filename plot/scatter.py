from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 0) 아티팩트 경로 자동 탐색
CAND = [Path("src/artifacts"), Path("artifacts")]
ART = next((c.resolve() for c in CAND if (c/"embeddings_train_X.npy").exists()), None)
if ART is None:
    raise FileNotFoundError(
        "embeddings_* 파일을 찾을 수 없습니다. 다음 폴더 중 하나에 두세요:\n"
        " - src/artifacts\n - artifacts\n"
        "필수 파일: embeddings_train_X.npy, embeddings_ood_X.npy"
    )
print("Using artifacts at:", ART)

# 1) 데이터 로드
X_ind = np.load(ART/"embeddings_train_X.npy")   # (N_ind, D)
X_ood = np.load(ART/"embeddings_ood_X.npy")     # (N_ood, D)

# 2) PCA 2D 투영
X_all = np.vstack([X_ind, X_ood])
Xc = X_all - X_all.mean(0, keepdims=True)
_, _, Vt = np.linalg.svd(Xc, full_matrices=False)
Z = Xc @ Vt[:2].T
Z_ind, Z_ood = Z[:len(X_ind)], Z[len(X_ind):]

# 3) 산점도
plt.figure(figsize=(8, 6))
plt.scatter(Z_ind[:,0], Z_ind[:,1], s=10, alpha=0.7, label="IND", c="#1f77b4")
plt.scatter(Z_ood[:,0], Z_ood[:,1], s=10, alpha=0.7, label="OOD", c="#d62728")
plt.title("IND vs OOD scatter (PCA)")
plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
plt.legend(); plt.tight_layout()

out = ART / "scatter.png"
plt.savefig(out, dpi=160)
print("저장:", out)
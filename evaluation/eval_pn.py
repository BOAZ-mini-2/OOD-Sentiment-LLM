from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, classification_report
)

# ----------------------------------------
# Config
# ----------------------------------------
GAMMA = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------
# artifact dir finder
# ----------------------------------------
def find_artifact_dir(
    candidates: Optional[List[Path]] = None,
    required_files: Optional[List[str]] = None
) -> Path:
    if candidates is None:
        candidates = [Path("src/artifacts"), Path("artifacts")]

    if required_files is None:
        required_files = ["embeddings_test_X.npy", "embeddings_test_Y.npy"]

    for cand in candidates:
        if all((cand / fname).exists() for fname in required_files):
            return cand.resolve()

    raise FileNotFoundError("필수 파일을 찾을 수 없음.")


# ----------------------------------------
# Projection Head (same as training)
# ----------------------------------------
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



# ----------------------------------------
# Prototype distance-based logits
# ----------------------------------------
def prototype_logits(fx, proto_neg, proto_pos, gamma=GAMMA):
    diff_neg = fx - proto_neg.unsqueeze(0)
    diff_pos = fx - proto_pos.unsqueeze(0)

    dist_neg = (diff_neg ** 2).sum(dim=1)
    dist_pos = (diff_pos ** 2).sum(dim=1)

    logits = torch.stack([-gamma * dist_neg, -gamma * dist_pos], dim=1)
    return logits


# ----------------------------------------
# Predict 함수 (ONLY Projection + Prototype)
# ----------------------------------------
@torch.no_grad()
def predict(model_proj, proto_neg, proto_pos, X: np.ndarray, device="cpu"):
    X_t = torch.from_numpy(X).float().to(device)

    fx = model_proj(X_t)
    logits = prototype_logits(fx, proto_neg, proto_pos, GAMMA)
    probs = torch.softmax(logits, dim=1).cpu().numpy()

    preds = probs.argmax(axis=1)
    return probs, preds


# ----------------------------------------
# MAIN
# ----------------------------------------
def main():
    ART = find_artifact_dir()
    print("Using artifacts at:", ART)

    # 1) Load Test set
    X_test = np.load(ART / "embeddings_test_X.npy")
    y_test = np.load(ART / "embeddings_test_Y.npy")
    print("Loaded:", X_test.shape, y_test.shape)

    in_dim = X_test.shape[1]

    # 2) Load ckpt
    ckpt_path = ART / "best_pn_classifier.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    proj_dim = ckpt.get("proj_dim", 256)

    # 3) Load ProjectionHead
    model_proj = ProjectionHead(in_dim=in_dim, proj_dim=proj_dim).to(DEVICE)
    model_proj.load_state_dict(ckpt["proj"])
    model_proj.eval()

    # 4) Load prototypes
    proto_neg = torch.tensor(ckpt["proto_neg"], dtype=torch.float32, device=DEVICE)
    proto_pos = torch.tensor(ckpt["proto_pos"], dtype=torch.float32, device=DEVICE)

    # optional: normalize proto (usually already normalized by EMA process)
    proto_neg = proto_neg / (proto_neg.norm() + 1e-12)
    proto_pos = proto_pos / (proto_pos.norm() + 1e-12)

    # 5) Predict
    probs, preds = predict(model_proj, proto_neg, proto_pos, X_test, DEVICE)

    # 6) Metrics
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary")
    cm = confusion_matrix(y_test, preds)

    print("\n[Metrics]")
    print("Accuracy :", f"{acc:.4f}")
    print("Precision:", f"{prec:.4f}")
    print("Recall   :", f"{rec:.4f}")
    print("F1       :", f"{f1:.4f}")
    print("CM:\n", cm)
    print("\nClassification report:\n", classification_report(y_test, preds, digits=4))

    try:
        auc = roc_auc_score(y_test, probs[:, 1])
        print("ROC-AUC:", f"{auc:.4f}")
    except:
        print("ROC-AUC 계산 실패")

    # Save summary
    summary_path = ART / "test_eval_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write("CM:\n")
        f.write(str(cm) + "\n")
        f.write("\nClassification report:\n")
        f.write(classification_report(y_test, preds, digits=4))

    print("\nSaved:", summary_path)


if __name__ == "__main__":
    main()




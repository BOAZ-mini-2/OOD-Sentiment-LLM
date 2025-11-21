from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, classification_report
)

# 0) 아티팩트 경로 자동 탐색 (src/artifacts → artifacts 순서)
def find_artifact_dir() -> Path:
    for cand in [Path("src/artifacts"), Path("artifacts")]:
        if (cand / "embeddings_test_X.npy").exists() and (cand / "embeddings_test_Y.npy").exists():
            return cand.resolve()
    raise FileNotFoundError(
        "테스트 임베딩 파일을 찾을 수 없습니다. 다음 폴더 중 한 곳에 두세요:\n"
        " - src/artifacts\n - artifacts\n"
        "필수 파일: embeddings_test_X.npy, embeddings_test_Y.npy"
    )

# 1) 학습 때와 동일한 모델 정의
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
    def forward(self, x): return self.net(x)

class ClassifierHead(nn.Module):
    def __init__(self, proj_dim: int, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(proj_dim, num_classes)
    def forward(self, fx): return self.fc(fx)

@torch.no_grad()
def predict(model_proj, model_clf, X: np.ndarray, device: str = "cpu"):
    X_t = torch.from_numpy(X).to(device).float()
    fx = model_proj(X_t)
    logits = model_clf(fx)            # (N, 2)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)
    return probs, preds

def main():
    ART = find_artifact_dir()
    print("Using artifacts at:", ART)

    # 2) 테스트 임베딩 로드 (라벨 파일 이름: embeddings_test_Y.npy ← 대문자 Y)
    #    y_test 값은 0(neg), 1(pos) 이어야 함
    X_test = np.load(ART / "embeddings_test_X.npy")   # (N, D)
    y_test = np.load(ART / "embeddings_test_Y.npy")   # (N,)
    in_dim  = X_test.shape[1]
    print(f"Loaded TEST: X_test={X_test.shape}, y_test={y_test.shape}")

    # 3) 모델/체크포인트 로드
    #    - proj_dim은 학습 때 값과 동일해야 함(기본 256로 학습)
    ckpt_path = ART / "best_pn_classifier.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"모델 체크포인트를 찾을 수 없습니다: {ckpt_path}\n"
            "학습 스크립트로 best_pn_classifier.pt를 먼저 생성해 주세요."
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 필요시 proj_dim을 ckpt에서 복원 (없으면 256 가정)
    proj_dim = ckpt.get("proj_dim", 256)
    model_proj = ProjectionHead(in_dim=in_dim, proj_dim=proj_dim)
    model_clf  = ClassifierHead(proj_dim=proj_dim, num_classes=2)
    model_proj.load_state_dict(ckpt["proj"])
    model_clf.load_state_dict(ckpt["clf"])
    model_proj.eval(); model_clf.eval()

    # 4) 예측 & 지표 출력
    probs, preds = predict(model_proj, model_clf, X_test, device="cpu")

    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="binary", pos_label=1
    )
    cm = confusion_matrix(y_test, preds)

    print("\n[Results]")
    print("Accuracy :", f"{acc:.4f}")
    print("Precision:", f"{prec:.4f} (positive=1)")
    print("Recall   :", f"{rec:.4f} (positive=1)")
    print("F1-score :", f"{f1:.4f} (positive=1)")
    print("\nConfusion Matrix (rows=true, cols=pred):\n", cm)
    print("\nClassification report:")
    print(classification_report(y_test, preds, digits=4))

    try:
        auc = roc_auc_score(y_test, probs[:, 1])
        print("ROC-AUC  :", f"{auc:.4f}")
    except Exception:
        print("ROC-AUC  : 계산 불가(레이블/확률 확인 필요)")

    # 선택: 결과 저장
    out = ART / "test_eval_summary.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision(pos=1): {prec:.4f}\n")
        f.write(f"Recall(pos=1): {rec:.4f}\n")
        f.write(f"F1(pos=1): {f1:.4f}\n")
        try:
            f.write(f"ROC-AUC: {auc:.4f}\n")
        except:
            pass
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")
        f.write("\nClassification report:\n")
        f.write(classification_report(y_test, preds, digits=4))
    print("\n요약 저장:", out)

if __name__ == "__main__":
    main()

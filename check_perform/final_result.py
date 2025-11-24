import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, balanced_accuracy_score,
    precision_recall_curve, average_precision_score
)

from ood_scoring.scoring import (
    score_msp,
    score_energy_from_probs,
    fit_md,
    score_md,
)
from check_perform.DMResult import DMResult



class OODEvaluator:
    """
    Full OOD evaluation wrapper for (X_val, y_val, probs_val).

    y_val: 0 = IND, 1 = OOD
    probs_val: softmax probs, shape (N, 2)
    """

    def __init__(self, X_val: np.ndarray, y_val: np.ndarray, probs_val: np.ndarray):
        self.X_val = np.asarray(X_val)
        self.y_val = np.asarray(y_val)
        self.probs_val = np.asarray(probs_val)

        # IND / OOD 분리
        self.X_IND = self.X_val[self.y_val == 0]
        self.X_OOD = self.X_val[self.y_val == 1]

        # scores & DMResult 저장용
        self.msp_scores = None
        self.energy_scores = None
        self.md_scores = None

        self.dm_msp: DMResult | None = None
        self.dm_energy: DMResult | None = None
        self.dm_md: DMResult | None = None

    # =========================================================
    # 1) Visualization
    # =========================================================
    def plot_latent_scatter(self, title="IND/OOD Latent (PCA 2D)"):
        """X_val 전체를 PCA 2D로 시각화, y_val로 색 구분"""
        from sklearn.decomposition import PCA

        z = PCA(n_components=2).fit_transform(self.X_val)

        IND = (self.y_val == 0)
        OOD = (self.y_val == 1)

        plt.figure(figsize=(8, 6))
        plt.scatter(z[IND, 0], z[IND, 1],
                    c="blue", s=10, alpha=0.6, label="IND")
        plt.scatter(z[OOD, 0], z[OOD, 1],
                    c="red", s=10, alpha=0.6, label="OOD")

        plt.title(title)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(alpha=0.2)
        plt.legend()
        plt.show()

    def plot_softmax_lines(self, title="Softmax Distribution (Real Model)"):
        """IND/OOD softmax 확률 시각화"""
        IND = (self.y_val == 0)
        OOD = (self.y_val == 1)

        probs_val = self.probs_val

        plt.figure(figsize=(8, 6))

        plt.plot(np.sort(probs_val[IND, 0]), label="IND: p_neg", color="blue")
        plt.plot(np.sort(probs_val[IND, 1]), label="IND: p_pos", color="blue", linestyle="--")

        plt.plot(np.sort(probs_val[OOD, 0]), label="OOD: p_neg", color="red")
        plt.plot(np.sort(probs_val[OOD, 1]), label="OOD: p_pos", color="red", linestyle="--")

        plt.title(title)
        plt.xlabel("Sorted index")
        plt.ylabel("Probability value")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.show()

    # =========================================================
    # 2) Summary
    # =========================================================
    def summarize_features(self):
        IND = self.X_IND
        OOD = self.X_OOD

        print("=== Feature Summary ===")
        print(f"IND count: {len(IND)}")
        print(f"OOD count: {len(OOD)}")
        print("\nIND mean (first 5 dims):", IND.mean(axis=0)[:5])
        print("OOD mean (first 5 dims):", OOD.mean(axis=0)[:5])

    def summarize_probs(self):
        probs_val = self.probs_val
        y_val = self.y_val

        print("\n=== Softmax Probability Summary ===")

        for lab, name in [(0, "IND"), (1, "OOD")]:
            idx = (y_val == lab)
            p = probs_val[idx]

            print(f"\n[{name}]")
            print(f"  Count: {idx.sum()}")
            print(f"  p_neg mean: {p[:,0].mean():.3f}, var: {p[:,0].var():.4f}")
            print(f"  p_pos mean: {p[:,1].mean():.3f}, var: {p[:,1].var():.4f}")

    # =========================================================
    # 3) Score 계산 + DMResult 평가
    # =========================================================
    def compute_scores(self, T: float = 1.0, reg_eps: float = 1e-5):
        """MSP, Energy, Mahalanobis score 계산 (값만 저장)"""
        # MSP, Energy
        self.msp_scores = score_msp(self.probs_val)
        self.energy_scores = score_energy_from_probs(self.probs_val, T=T)

        # MD: IND 만으로 fitting
        mu_md, inv_cov_md = fit_md(self.X_IND, reg_eps=reg_eps)
        self.md_scores = score_md(self.X_val, mu_md, inv_cov_md)

    def evaluate_all(self, target_tpr: float = 0.95):
        """
        DMResult 3개(MSP, Energy, MD)를 계산해서 저장.
        compute_scores()가 먼저 호출되어 있어야 함.
        target_tpr는 FPR@target_tpr (예: FPR95) 계산용으로만 쓰임.
        """
        if self.msp_scores is None or self.energy_scores is None or self.md_scores is None:
            raise RuntimeError("compute_scores()를 먼저 호출하세요.")

        print("\n=== MSP OOD Performance ===")
        self.dm_msp = DMResult()(self.y_val, self.msp_scores, target_tpr=target_tpr)
        self.dm_msp.summary()

        print("\n=== Energy OOD Performance ===")
        self.dm_energy = DMResult()(self.y_val, self.energy_scores, target_tpr=target_tpr)
        self.dm_energy.summary()

        print("\n=== Mahalanobis OOD Performance ===")
        self.dm_md = DMResult()(self.y_val, self.md_scores, target_tpr=target_tpr)
        self.dm_md.summary()

    # =========================================================
    # 4) ROC / PR plotting (각 measure 별로 호출)
    # =========================================================
    def plot_roc_msp(self):
        if self.dm_msp is None:
            raise RuntimeError("evaluate_all() 이후에 호출하세요.")
        self.dm_msp.plot_roc("ROC Curve (MSP)")

    def plot_pr_msp(self):
        if self.dm_msp is None:
            raise RuntimeError("evaluate_all() 이후에 호출하세요.")
        self.dm_msp.plot_pr("PR Curve (MSP)")

    def plot_roc_energy(self):
        if self.dm_energy is None:
            raise RuntimeError("evaluate_all() 이후에 호출하세요.")
        self.dm_energy.plot_roc("ROC Curve (Energy)")

    def plot_pr_energy(self):
        if self.dm_energy is None:
            raise RuntimeError("evaluate_all() 이후에 호출하세요.")
        self.dm_energy.plot_pr("PR Curve (Energy)")

    def plot_roc_md(self):
        if self.dm_md is None:
            raise RuntimeError("evaluate_all() 이후에 호출하세요.")
        self.dm_md.plot_roc("ROC Curve (Mahalanobis)")

    def plot_pr_md(self):
        if self.dm_md is None:
            raise RuntimeError("evaluate_all() 이후에 호출하세요.")
        self.dm_md.plot_pr("PR Curve (Mahalanobis)")




def test_ood_detection(
    X_test,
    y_test,
    probs_test,
    best_name: str,
    best_thr: float,
    T: float = 1.0,
    reg_eps: float = 1e-5,
):
    """
    테스트 셋에서 OOD 성능 평가 (threshold는 validation에서 가져온 값 사용)

    Parameters
    ----------
    X_test : np.ndarray, shape (N, D)
        테스트 임베딩
    y_test : array-like, shape (N,)
        0 = IND, 1 = OOD
    probs_test : np.ndarray, shape (N, 2)
        테스트 softmax 확률
    best_name : {"MSP", "Energy", "MD"}
        validation 단계에서 선택된 best measure 이름
    best_thr : float
        validation 단계에서 해당 measure 기준으로 뽑은 threshold
    T : float, default=1.0
        Energy score 계산에 사용하는 온도
    reg_eps : float, default=1e-5
        MD 계산 시 covariance regularization

    Returns
    -------
    None
        성능을 보기 좋게 print만 함.
    """

    best_name = best_name.upper()
    y_test = np.asarray(y_test)
    X_test = np.asarray(X_test)
    probs_test = np.asarray(probs_test)

    # ------------------------------------
    # 1) measure에 맞는 score 재계산
    # ------------------------------------
    if best_name == "MSP":
        scores = score_msp(probs_test)

    elif best_name == "ENERGY":
        scores = score_energy_from_probs(probs_test, T=T)

    elif best_name == "MD":
        # 테스트 IND만으로 MD fit (혹은 원래 train IND를 따로 넘기도록 바꿔도 됨)
        X_IND_test = X_test[y_test == 0]
        mu_md, inv_cov_md = fit_md(X_IND_test, reg_eps=reg_eps)
        scores = score_md(X_test, mu_md, inv_cov_md)

    else:
        raise ValueError("best_name 은 'MSP', 'Energy', 'MD' 중 하나여야 합니다.")

    # ------------------------------------
    # 2) threshold 적용 → 예측 라벨
    # ------------------------------------
    y_pred = (scores >= best_thr).astype(int)   # 1 = OOD, 0 = IND

    # ------------------------------------
    # 3) 혼동행렬 기반 지표
    # ------------------------------------
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    TPR = tp / (tp + fn + 1e-12)   # OOD recall
    TNR = tn / (tn + fp + 1e-12)   # IND recall
    bal_acc = (TPR + TNR) / 2

    # ------------------------------------
    # 4) AUPR (threshold-free)
    # ------------------------------------
    aupr = average_precision_score(y_test, scores)

    # ------------------------------------
    # 5) 출력
    # ------------------------------------
    print("\n========================================")
    print(f"  Test OOD Detection Performance ({best_name})")
    print("========================================")
    print(f"Threshold (from validation): {best_thr:.6f}\n")
    print(f"TPR (Recall OOD) : {TPR:.4f}")
    print(f"TNR (Recall IND) : {TNR:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"AUPR             : {aupr:.4f}")
    print("========================================\n")

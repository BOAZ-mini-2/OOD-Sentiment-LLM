import numpy as np
import matplotlib.pyplot as plt

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

    # =========================================================
    # 5) FPR95 기준 best measure + threshold 반환
    #    (지금은 DMResult의 fpr95/get_trs 기준 그대로 사용)
    # =========================================================
    def get_best_by_fpr95(self, target_tpr: float = 0.95):
        """
        DMResult의 fpr95를 비교해서
        FPR95가 가장 낮은 measure와 threshold를 반환한다.

        return:
            best_name, best_thr, best_fpr
        """
        if any(dm is None for dm in [self.dm_msp, self.dm_energy, self.dm_md]):
            raise RuntimeError("evaluate_all() 이후에 호출하세요.")

        fprs = {
            "MSP":    self.dm_msp.fpr95,
            "Energy": self.dm_energy.fpr95,
            "MD":     self.dm_md.fpr95
        }

        best_name = min(fprs, key=fprs.get)
        best_fpr = fprs[best_name]

        # threshold는 DMResult.get_trs 기준 (TPR>=target_tpr)
        if best_name == "MSP":
            best_thr = self.dm_msp.get_trs(target_tpr)
        elif best_name == "Energy":
            best_thr = self.dm_energy.get_trs(target_tpr)
        else:
            best_thr = self.dm_md.get_trs(target_tpr)

        return best_name, best_thr, best_fpr

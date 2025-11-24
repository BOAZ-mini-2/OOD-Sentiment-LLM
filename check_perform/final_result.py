# ============================================================
# Full OOD evaluation with visualization (Real IND/OOD data)
# Using X_val, y_val, probs_val
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from ood_scoring.scoring import (
    score_msp,
    score_energy_from_probs,
    fit_md,
    score_md,
)
from check_perform.DMResult import DMResult


# ============================================================
# 1) Visualization helpers (PCA + Softmax)
# ============================================================

def plot_latent_scatter_real_val(X_val, y_val, title="IND/OOD Latent (PCA 2D)"):
    """X_val 전체를 PCA 2D로 시각화, y_val로 색 구분"""

    from sklearn.decomposition import PCA

    z = PCA(n_components=2).fit_transform(X_val)

    IND = (y_val == 0)
    OOD = (y_val == 1)

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


def plot_softmax_lines_real_val(probs_val, y_val, title="Softmax Distribution (Real Model)"):
    """IND/OOD softmax 확률 시각화"""

    IND = (y_val == 0)
    OOD = (y_val == 1)

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


# ============================================================
# 2) Summary helpers
# ============================================================

def summarize_features_real_val(X_val, y_val):
    IND = X_val[y_val == 0]
    OOD = X_val[y_val == 1]

    print("=== Feature Summary ===")
    print(f"IND count: {len(IND)}")
    print(f"OOD count: {len(OOD)}")
    print("\nIND mean (first 5 dims):", IND.mean(axis=0)[:5])
    print("OOD mean (first 5 dims):", OOD.mean(axis=0)[:5])


def summarize_probs_real_val(probs_val, y_val):
    print("\n=== Softmax Probability Summary ===")

    for lab, name in [(0, "IND"), (1, "OOD")]:
        idx = (y_val == lab)
        p = probs_val[idx]

        print(f"\n[{name}]")
        print(f"  Count: {idx.sum()}")
        print(f"  p_neg mean: {p[:,0].mean():.3f}, var: {p[:,0].var():.4f}")
        print(f"  p_pos mean: {p[:,1].mean():.3f}, var: {p[:,1].var():.4f}")


# ============================================================
# 3) Main OOD routine (X_val, y_val 전용)
# ============================================================

def run_ood_full_val(X_val, y_val, probs_val):
    print(">>> Running OOD Evaluation (X_val / y_val Version) <<<")

    X_IND = X_val[y_val == 0]
    X_OOD = X_val[y_val == 1]

    # ============================
    # Visualization
    # ============================
    plot_latent_scatter_real_val(X_val, y_val)
    plot_softmax_lines_real_val(probs_val, y_val)

    # ============================
    # Summary
    # ============================
    summarize_features_real_val(X_val, y_val)
    summarize_probs_real_val(probs_val, y_val)

    # ============================
    # OOD Scores
    # ============================
    print("\n--- OOD Scoring ---")

    msp_scores = score_msp(probs_val)
    energy_scores = score_energy_from_probs(probs_val, T=1.0)

    mu_md, inv_cov_md = fit_md(X_IND, reg_eps=1e-5)
    md_scores = score_md(X_val, mu_md, inv_cov_md)

    # ============================
    # DMResult for all measures
    # ============================
    print("\n=== MSP OOD Performance ===")
    dm_msp = DMResult()(y_val, msp_scores)
    dm_msp.summary()
    dm_msp.plot_roc("ROC Curve (MSP)")
    dm_msp.plot_pr("PR Curve (MSP)")

    print("\n=== Energy OOD Performance ===")
    dm_energy = DMResult()(y_val, energy_scores)
    dm_energy.summary()
    dm_energy.plot_roc("ROC Curve (Energy)")
    dm_energy.plot_pr("PR Curve (Energy)")

    print("\n=== Mahalanobis OOD Performance ===")
    dm_md = DMResult()(y_val, md_scores)
    dm_md.summary()
    dm_md.plot_roc("ROC Curve (Mahalanobis)")
    dm_md.plot_pr("PR Curve (Mahalanobis)")

    # ============================
    # BEST measure selection (by FPR95)
    # ============================
    fprs = {
        "MSP":    dm_msp.fpr95,
        "Energy": dm_energy.fpr95,
        "MD":     dm_md.fpr95
    }

    best_measure = min(fprs, key=fprs.get)
    best_fpr = fprs[best_measure]

    print("\n====================================================")
    print(f"Best measure (lowest FPR95): {best_measure}  (FPR95={best_fpr:.4f})")

    # pick threshold based on best measure
    if best_measure == "MSP":
        best_thr = dm_msp.get_trs(0.95)
    elif best_measure == "Energy":
        best_thr = dm_energy.get_trs(0.95)
    else:
        best_thr = dm_md.get_trs(0.95)

    print(f"Threshold @ TPR>=0.95 (for {best_measure}): {best_thr:.6f}")
    print("====================================================")



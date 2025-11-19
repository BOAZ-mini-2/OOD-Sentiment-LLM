'''
본 코드는 우리 프로젝트 데이터셋과 비슷하게 꾸민 random sample을 이용해
score를 구하고(from scoring) evaluate하는(from DMResult) 코드...
코드 다 짠거
GPT한테 뭉쳐달라해서 정리 XXX
'''
# ============================================================
# demo_fake_data.py
# Synthetic sentiment-like data + visualization + OOD scoring
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
# 1) Synthetic sentiment-like data generator
#    - labels: 0=pos, 1=neg, 2=near-OOD(neutral)
# ============================================================

def make_synthetic_sentiment_data(
    n_neg: int = 1000,
    n_pos: int = 1000,
    n_neu: int = 500,
    dim: int = 16,
    seed: int = 42,
    return_latent: bool = False,
):
    rng = np.random.default_rng(seed)

    # ---------- (A) 2D latent 구조 설계 ----------
    # neg: 왼쪽
    z_neg = rng.normal(loc=(-3.0, 0.0), scale=(0.7, 0.7), size=(n_neg, 2))
    # pos: 오른쪽
    z_pos = rng.normal(loc=(+3.0, 0.0), scale=(0.7, 0.7), size=(n_pos, 2))

    # near-OOD: 가운데 띠 (중간에 길게 퍼진 형상)
    z_neu = rng.normal(loc=(0.0, 0.0), scale=(1.5, 0.4), size=(n_neu, 2))
    # pos/neg 방향으로 살짝 이동 → 일부가 양쪽 클러스터에 스며듦
    direction = rng.choice([-1.0, +1.0], size=(n_neu, 1))
    z_neu[:, 0:1] += direction * rng.normal(0.8, 0.3, size=(n_neu, 1))

    z = np.vstack([z_neg, z_pos, z_neu])  # (N, 2)
    N = z.shape[0]

    # ---------- (B) latent 2D → 16D 임베딩 ----------
    W = rng.normal(0.0, 1.0, size=(2, dim))   # (2, D)
    feats = z @ W + rng.normal(0.0, 0.1, size=(N, dim))

    # ---------- (C) softmax 확률 (2-class: neg / pos) ----------
    probs_neg = np.stack([
        rng.uniform(0.8, 0.95, size=n_neg),   # p_neg ↑
        rng.uniform(0.05, 0.2, size=n_neg),   # p_pos ↓
    ], axis=1)

    probs_pos = np.stack([
        rng.uniform(0.05, 0.2, size=n_pos),   # p_neg ↓
        rng.uniform(0.8, 0.95, size=n_pos),   # p_pos ↑
    ], axis=1)

    probs_neu = np.stack([
        rng.uniform(0.45, 0.55, size=n_neu),  # 애매
        rng.uniform(0.45, 0.55, size=n_neu),
    ], axis=1)

    probs = np.vstack([probs_neg, probs_pos, probs_neu])  # (N, 2)

    # ---------- (D) labels: 0=pos, 1=neg, 2=near-OOD ----------
    labels = np.concatenate([
        np.ones(n_neg, dtype=int),        # neg = 1
        np.zeros(n_pos, dtype=int),       # pos = 0
        np.full(n_neu, 2, dtype=int),     # near-OOD = 2
    ])

    if return_latent:
        return feats, probs, labels, z
    return feats, probs, labels


# ============================================================
# 2) Visualization helpers
# ============================================================

def plot_latent_scatter(z, labels, title="Synthetic Sentiment Latent (2D)"):
    label_names = {0: "Positive", 1: "Negative", 2: "Neutral (near-OOD)"}
    colors = {0: "blue", 1: "red", 2: "green"}

    plt.figure(figsize=(8, 6))
    for lab in np.unique(labels):
        idx = labels == lab
        plt.scatter(
            z[idx, 0], z[idx, 1],
            s=10, alpha=0.7,
            c=colors[lab],
            label=label_names[lab],
        )

    plt.axvline(0, color="gray", linestyle=":", alpha=0.3)  # pos/neg 중간선
    plt.legend()
    plt.title(title)
    plt.xlabel("latent x")
    plt.ylabel("latent y")
    plt.grid(alpha=0.2)
    plt.show()


def plot_softmax_lines(probs, labels, title="Softmax Probability Distribution"):
    label_names = {0: "Positive", 1: "Negative", 2: "Neutral (near-OOD)"}
    colors = {0: "blue", 1: "red", 2: "green"}

    plt.figure(figsize=(8, 6))

    for lab in np.unique(labels):
        idx = labels == lab
        p_neg = probs[idx, 0]
        p_pos = probs[idx, 1]

        plt.plot(
            np.sort(p_neg),
            label=f"{label_names[lab]}: p_neg",
            color=colors[lab],
            linestyle="-",
        )
        plt.plot(
            np.sort(p_pos),
            label=f"{label_names[lab]}: p_pos",
            color=colors[lab],
            linestyle="--",
        )

    plt.title(title)
    plt.xlabel("Sorted sample index")
    plt.ylabel("Probability value")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()


# ============================================================
# 3) Summary helpers
# ============================================================

def summarize_features(feats, labels):
    """
    feats: (N, D)
    labels: (N,) with 0=pos, 1=neg, 2=near-OOD
    """
    classes = {0: "Positive", 1: "Negative", 2: "Neutral (near-OOD)"}

    D = feats.shape[1]
    print("=== Feature Summary ({} dims) ===".format(D))

    for lab in [0, 1, 2]:
        idx = labels == lab
        feats_lab = feats[idx]

        mean = feats_lab.mean(axis=0)
        var = feats_lab.var(axis=0)

        print(f"\n[{classes[lab]}]")
        print(f"  Count: {idx.sum()}")
        print(f"  Mean (first 5 dims): {mean[:5]}")
        print(f"  Var  (first 5 dims): {var[:5]}")
        print(f"  Mean L2-norm: {np.linalg.norm(mean):.3f}")
        print(f"  Avg variance: {var.mean():.3f}")


def summarize_probs(probs, labels):
    """
    probs: (N, 2) = [p_neg, p_pos]
    labels: (N,)
    """
    classes = {0: "Positive", 1: "Negative", 2: "Neutral (near-OOD)"}

    print("\n=== Softmax Probability Summary ===")

    for lab in [0, 1, 2]:
        idx = labels == lab
        p = probs[idx]

        p_neg = p[:, 0]
        p_pos = p[:, 1]

        print(f"\n[{classes[lab]}]")
        print(f"  Count: {idx.sum()}")
        print(f"  p_neg mean: {p_neg.mean():.3f}, var: {p_neg.var():.4f}")
        print(f"  p_pos mean: {p_pos.mean():.3f}, var: {p_pos.var():.4f}")
        print(f"  p_neg range: ({p_neg.min():.3f}, {p_neg.max():.3f})")
        print(f"  p_pos range: ({p_pos.min():.3f}, {p_pos.max():.3f})")


# ============================================================
# 4) Main: scoring + DMResult OOD evaluation
# ============================================================

def main():
    print(">>> Synthetic sentiment demo with MSP / Energy / MD + DMResult <<<")

    # ---- 1) 데이터 생성 ----
    feats, probs, labels, z = make_synthetic_sentiment_data(return_latent=True)

    print("feats shape:", feats.shape)
    print("probs shape:", probs.shape)
    print("labels shape:", labels.shape)

    # ---- 2) 시각화 + summary (옵션이지만 지금은 항상 호출) ----
    plot_latent_scatter(z, labels)
    plot_softmax_lines(probs, labels)

    summarize_features(feats, labels)
    summarize_probs(probs, labels)

    # ---- 3) IND vs near-OOD 라벨 생성 (DMResult용) ----
    #   y_true: 0 = IND(pos/neg), 1 = OOD(neutral)
    is_ood = (labels == 2).astype(int)

    # ---- 4) MD 학습/평가용 train/test split ----
    rng = np.random.default_rng(0)
    ind_idx = np.where(labels != 2)[0]
    ood_idx = np.where(labels == 2)[0]

    rng.shuffle(ind_idx)
    rng.shuffle(ood_idx)

    n_ind = len(ind_idx)
    n_train_ind = int(0.7 * n_ind)

    train_idx = ind_idx[:n_train_ind]                    # IND 중 70%만 train
    test_idx = np.concatenate([ind_idx[n_train_ind:],    # 나머지 IND
                               ood_idx])                 # + 모든 OOD

    feats_train = feats[train_idx]
    feats_test = feats[test_idx]
    probs_test = probs[test_idx]
    y_true = is_ood[test_idx]

    print("\nTrain feats:", feats_train.shape)
    print("Test feats:", feats_test.shape)
    print("Test probs:", probs_test.shape)
    print("y_true (0=IND,1=OOD) shape:", y_true.shape)

    # ---- 5) Scoring: MSP / Energy / MD ----
    # MSP: 클수록 OOD 라는 convention으로 구현돼 있다고 가정
    msp_scores = score_msp(probs_test)
    print("\nMSP scores shape:", msp_scores.shape)
    print("MSP scores example:", msp_scores[:5])

    # Energy (probs 기반): 클수록 OOD 로 해석
    energy_scores = score_energy_from_probs(probs_test, T=1.0)
    print("\nEnergy scores shape:", energy_scores.shape)
    print("Energy scores example:", energy_scores[:5])

    # Mahalanobis: IND train feats로 fitting 후, 거리 클수록 OOD
    mu_md, inv_cov_md = fit_md(feats_train, reg_eps=1e-5)
    md_scores = score_md(feats_test, mu_md, inv_cov_md)
    print("\nMD scores shape:", md_scores.shape)
    print("MD scores example:", md_scores[:5])

    # ---- 6) DMResult로 OOD 성능 평가 ----
    # DMResult 가정:
    #   y_true: 0=IND, 1=OOD
    #   scores: 클수록 OOD-like

    print("\n=== MSP OOD metrics ===")
    dm_msp = DMResult()(y_true, msp_scores)
    dm_msp.summary()

    print("\n=== Energy OOD metrics ===")
    dm_energy = DMResult()(y_true, energy_scores)
    dm_energy.summary()

    print("\n=== MD OOD metrics ===")
    dm_md = DMResult()(y_true, md_scores)
    dm_md.summary()


# ============================================================
# 5) Entry point
# ============================================================

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("TkAgg")
    main()


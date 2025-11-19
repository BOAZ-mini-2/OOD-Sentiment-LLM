# ood_scoring/examples/demo_fake_data.py

import numpy as np
from ood_scoring.scoring import (
    score_msp,
    score_energy_from_probs,
    fit_md,
    score_md,
)

from check_perform.DMResult import DMResult

def main():
    np.random.seed(42)

    N = 1000    # samples
    C = 2       # class
    D = 16      # feature dim

    # 1) logits -> probs 생성
    logits = np.random.randn(N, C)
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)  # (N, C)

    # 2) feature 생성
    feats = np.random.randn(N, D)  # (N, D)

    # train / test 분리 (MD)
    n_train = N // 2
    feats_train = feats[:n_train]
    feats_test = feats[n_train:]
    probs_test = probs[n_train:]

    print("probs shape:", probs.shape)
    print("feats shape:", feats.shape)
    print("probs_test shape:", probs_test.shape)
    print("feats_train shape:", feats_train.shape)
    print("feats_test shape:", feats_test.shape)

    # 3) post-hoc scoring
    # ===== MSP =====
    msp_scores = score_msp(probs_test)
    print("\nMSP scores shape:", msp_scores.shape)
    print("MSP scores example:", msp_scores[:5])

    # ===== Energy (from probs) =====
    energy_scores = score_energy_from_probs(probs_test, T=1.0)
    print("\nEnergy scores shape:", energy_scores.shape)
    print("Energy scores example:", energy_scores[:5])

    # ===== Mahalanobis =====
    mu_md, inv_cov_md = fit_md(feats_train, reg_eps=1e-5)
    md_scores = score_md(feats_test, mu_md, inv_cov_md)
    print("\nMD scores shape:", md_scores.shape)
    print("MD scores example:", md_scores[:5])
    
    # 4) evaluate
    # DMResult.py 가정:
    #   y_true: 0 = IND, 1 = OOD
    #   scores: 클수록 OOD-like

    N_test = probs_test.shape[0]

    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=N_test)   # 0 또는 1

    # ===== MSP =====
    dm_msp = DMResult()(y_true, msp_scores)
    print("\n=== MSP OOD metrics ===")
    dm_msp.summary()

    # ===== E =====
    dm_energy = DMResult()(y_true, energy_scores)
    print("\n=== Energy OOD metrics ===")
    dm_energy.summary()

    # ===== MD     =====
    dm_md = DMResult()(y_true, md_scores)
    print("\n=== MD OOD metrics ===")
    dm_md.summary()

if __name__ == "__main__":
    main()

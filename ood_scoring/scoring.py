# ood_scoring/scoring.py
'''
from ood_scoring.scoring import (
    score_msp,
    score_energy_from_probs,
    score_energy_from_logits,
    fit_md,
    score_md,
)
'''

import numpy as np

## 1. MSP-based OOD score

def score_msp(probs: np.ndarray) -> np.ndarray:
    '''
    input :
    -> probs : np.ndarray
    -> shape (N, C)
    -> softmax 확률값

    ouput :
    -> scores = 1.0 - max_prob
    -> shape (N,)
    -> 클수록 OOD
    '''
    if probs.ndim != 2:
        raise ValueError(f"`probs` must be 2D (N, C), got shape {probs.shape}")

    max_prob = probs.max(axis=1)
    scores = 1.0 - max_prob
    return scores

## 2. Energy-baed OOD score

def score_energy_from_probs(probs: np.ndarray, T: float = 1.0) -> np.ndarray:
    # probs 기반 energy scoring
    '''
    input :
    -> probs : np.ndarray, T : float
    -> shape (N, C)
    -> Temperature (default 1.0)

    ouput :
    -> scores = T * np.log(sum_exp + 1e-12)
    -> shape (N,)
    -> 클수록 OOD
    '''
    if probs.ndim != 2:
        raise ValueError(f"`probs` must be 2D (N, C), got shape {probs.shape}")
    if T <= 0:
        raise ValueError("T must be positive.")

    scaled = probs / T
    exp_scaled = np.exp(scaled)
    sum_exp = exp_scaled.sum(axis=1)
    scores = T * np.log(sum_exp + 1e-12)
    return scores

def score_energy_from_logits(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    # logit 기반 energy scoring
    if logits.ndim != 2:
        raise ValueError(f"`logits` must be 2D (N, C), got shape {logits.shape}")
    if T <= 0:
        raise ValueError("T must be positive.")

    scaled = logits / T
    max_scaled = scaled.max(axis=1, keepdims=True)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = exp_scaled.sum(axis=1)
    log_sum_exp = max_scaled.squeeze(axis=1) + np.log(sum_exp + 1e-12)
    scores = T * log_sum_exp
    return scores

## 3. MD-based OOD score

def fit_md(feats_train: np.ndarray, reg_eps: float = 1e-5):
    # MD를 구하기 위한 parameter estimation fn
    '''
    input :
    -> feats_train : np.ndarray
    -> shape (N_train, D)
    -> In-distribution training set feature vector

    ouput :
    -> mu = feats_train.mean(axis=0), inv_cov = np.linalg.pinv(cov)
    -> shape (D,), shape (D, D)
    -> train IND feature의 평균 벡터, 역공분산 행렬
    '''
    if feats_train.ndim != 2:
        raise ValueError(f"`feats_train` must be 2D (N_train, D), got shape {feats_train.shape}")

    mu = feats_train.mean(axis=0)
    cov = np.cov(feats_train, rowvar=False)

    if reg_eps is not None and reg_eps > 0:
        cov = cov + reg_eps * np.eye(cov.shape[0])

    inv_cov = np.linalg.pinv(cov)
    return mu, inv_cov

def score_md(feats: np.ndarray, mu: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    '''
    input :
    -> feats : np.ndarray, mu, inv_cov
    -> shape (N, D)
    -> test에서 추출된 features, 평균, 역공분산

    output :
    -> scores = -np.sqrt(d_squared)
    -> Mahalanobis distance 값
    -> 작을수록 OOD
    '''
    if feats.ndim != 2:
        raise ValueError(f"`feats` must be 2D (N, D), got shape {feats.shape}")
    if mu.ndim != 1:
        raise ValueError(f"`mu` must be 1D (D,), got shape {mu.shape}")
    if inv_cov.ndim != 2:
        raise ValueError(f"`inv_cov` must be 2D (D, D), got shape {inv_cov.shape}")

    diff = feats - mu
    left = diff @ inv_cov
    d_squared = np.sum(left * diff, axis=1)
    d_squared = np.maximum(d_squared, 0.0)
    scores = -np.sqrt(d_squared)
    return scores

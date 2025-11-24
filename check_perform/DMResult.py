import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, average_precision_score,
    precision_recall_curve
)

class DMResult:
    """
    Distance-based OOD Metrics Container
    ====================================
    y_true: 0 = IND, 1 = OOD
    scores: higher value ⇒ more OOD-like
    """

    def __init__(self):
        self.fpr: np.ndarray = np.array([], dtype=float)
        self.tpr: np.ndarray = np.array([], dtype=float)
        self.thr: np.ndarray = np.array([], dtype=float)

        self.auroc: float = float('nan')
        self.fpr95: float = float('nan')
        self.aupr: float = float('nan')

    def __call__(self, y_true, scores,
                 pos_label: int = 1,
                 target_tpr: float = 0.95):

        y_true_arr = np.asarray(y_true)
        scores_arr = np.asarray(scores)

        # ROC curve
        self.fpr, self.tpr, self.thr = roc_curve(
            y_true_arr, scores_arr, pos_label=pos_label
        )

        # AUROC
        self.auroc = float(auc(self.fpr, self.tpr))

        # FPR@target_tpr
        idx = np.where(self.tpr >= target_tpr)[0]
        self.fpr95 = float(self.fpr[idx[0]]) if len(idx) > 0 else 1.0

        # AUPR
        self.aupr = float(average_precision_score(y_true_arr, scores_arr))

        # store for PR curve plotting
        self._y_true = y_true_arr
        self._scores = scores_arr

        return self

    def summary(self):
        print(f"AUROC : {self.auroc:.4f}")
        print(f"AUPR  : {self.aupr:.4f}")
        print(f"FPR95 : {self.fpr95:.4f}")
        print("========================")

    # ------------------------------------------------------------
    # (1) Threshold at TPR >= target_tpr   (ex: TPR=0.95)
    # ------------------------------------------------------------
    def get_trs(self, target_tpr: float = 0.95):
        """
        Returns the threshold where TPR first reaches 'target_tpr'.
        (Same threshold used for FPR95 when target_tpr=0.95)
        """
        if self.fpr.size == 0:
            raise RuntimeError("Call DMResult(y_true, scores) before get_trs().")

        idx = np.where(self.tpr >= target_tpr)[0]
        if len(idx) == 0:
            return float(self.thr[-1])  # fallback: highest threshold

        return float(self.thr[idx[0]])

    # ------------------------------------------------------------
    # (2) ROC curve plot
    # ------------------------------------------------------------
    def plot_roc(self, title="ROC Curve"):
        if self.fpr.size == 0:
            raise RuntimeError("Call DMResult(y_true, scores) before plot_roc().")

        plt.figure(figsize=(6, 5))
        plt.plot(self.fpr, self.tpr, label=f"AUROC={self.auroc:.4f}")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.4)

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.grid(alpha=0.2)
        plt.legend()
        plt.show()

    # ------------------------------------------------------------
    # (3) Precision-Recall curve plot
    # ------------------------------------------------------------
    def plot_pr(self, title="Precision-Recall Curve"):
        if not hasattr(self, "_y_true"):
            raise RuntimeError("Call DMResult(y_true, scores) before plot_pr().")

        precision, recall, _ = precision_recall_curve(self._y_true, self._scores)

        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, label=f"AUPR={self.aupr:.4f}")

        plt.xlabel("Recall (TPR)")
        plt.ylabel("Precision (PPV)")
        plt.title(title)
        plt.grid(alpha=0.2)
        plt.legend()
        plt.show()


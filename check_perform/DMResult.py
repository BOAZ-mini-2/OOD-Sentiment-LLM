import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc

class DMResult:
    """
    Distance-based OOD Metrics Container
    ====================================

    COMMENT
    --------
    Assumptions
    COMMENT
        * y_true: 0 = IND, 1 = OOD
        * scores: higher value ⇒ more OOD-like
        * ROC rule: score >= threshold → predicted OOD

    COMMENT
    --------
    ROC Components
    COMMENT
        fpr : False Positive Rate array
        tpr : True Positive Rate array
        thr : Threshold array  
               (fpr[k], tpr[k], thr[k]) correspond to the same operating point

    COMMENT
    --------
    Metrics
    COMMENT
        auroc : Area under ROC curve  
        aupr  : Area under Precision-Recall curve  
        fpr95 : FPR when TPR first reaches 0.95
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

        # ROC
        self.fpr, self.tpr, self.thr = roc_curve(
            y_true_arr, scores_arr, pos_label=pos_label
        )

        # AUROC
        self.auroc = float(auc(self.fpr, self.tpr))

        # FPR@95%TPR
        idx = np.where(self.tpr >= target_tpr)[0]
        self.fpr95 = float(self.fpr[idx[0]]) if len(idx) > 0 else 1.0

        # AUPR
        self.aupr = float(average_precision_score(y_true_arr, scores_arr))

        return self

    def summary(self):
        print(f"AUROC : {self.auroc:.4f}")
        print(f"AUPR  : {self.aupr:.4f}")
        print(f"FPR95 : {self.fpr95:.4f}")
        print("========================")

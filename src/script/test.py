# 지수가 그냥 혼자 짜본 draft인데 미프 회의하면서 또 구조가 바뀌어서,,, 그냥 참고용으로 유지해놓을게요

import os, random, math
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# 1) 데이터 준비
# 'IMDb': IND -> labeled(P/N)
# 'un-review data': OOD dataset -> unlabeled

def load_ds(n_per_class=1500, n_ood=3000):
    imdb = load_dataset("imdb")
    # train/test dataset 섞어서 sampling할 예정
    pass


# 2) mDeBERTa embedding (문장 임베딩)
# model : microsoft/mdeberta-v3-base (DeBERTa v3)

class MDebertaEmbedder:
    def __init__(self, model_name="microsoft/mdeberta-v3-base", device=None):
        pass

    def encoder():
        pass


# 3) LLaMA-based filtering (for step 2)

LLAMA_PROMPTS = {} # 어쩌구


class LlamaJudge:
    def __init__():
        pass

    # @torch.no_grad()
    # 더 논의 후 구현
    pass


# 4) 차원 축소 & plot
# parameter는 임의로 샘플 그대로 고정해놨고 나중에 바꿀 예정 ㅇㅇ
def project_2d(X, method="umap", random_state=SEED):
    Xz = StandardScaler().fit_transform(X) # 표준화
    if method == "umap":
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.05, metric="cosine", random_state=random_state)
        Z = reducer.fit_trainsform(Xz)
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=40, learning_rate=500, n_iter=2000, random_state=random_state)
        Z = reducer.fit_trainsform(Xz)
    return Z

def plot_scatter(Z, colors, markers, title):
    # plt.figure(figsize=(7, 2, 6))
    # for (c,m) in sorted(set(zip(colors, markers))):
    #     idx = [i for i,(cc,mm) in enumerate(zip(colors,markers)) if (cc==c and mm==m)]
    #     plt.scatter(Z[idx,0], Z[idx,1], s=9, alpha=0.75, c=c, marker=m, label=f"{c}-{m}")
    # plt.legend(loc="best", fontsize=8)
    # plt.title(title); plt.xlabel("dim-1"); plt.ylabel("dim-2")
    # plt.tight_layout(); plt.show()
    pass


# 5) main 함수
def main():
    pass



if __name__ == "__main__":
    main()




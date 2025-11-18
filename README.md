# OOD-Sentiment-LLM
> LLM-Based OOD Detection in Sentiment Classification
```python
git clone https://github.com/BOAZ-mini-2/OOD-Sentiment-LLM.git
cd OOD-Sentiment-LLM
```

### Our workflow draft
<img width="2256" height="697" alt="Image" src="https://github.com/user-attachments/assets/074effb8-30a1-4e33-98fb-e600f64efbc1" />

### Load dataset
```python
file_names = [
    "src/dataset/All_Beauty.jsonl_25k.jsonl",
    "src/dataset/Baby_Products.jsonl_25k.jsonl",
    "src/dataset/Grocery_and_Gourmet_Food.jsonl_25k.jsonl",
    "src/dataset/Industrial_and_Scientific.jsonl_25k.jsonl"
]
```

### Scoring method (MSP / E / MD)

$score_{MSP}(x) = 1 - \max_{y \in \mathcal{Y}} p(y \mid x)$
$score_{E}(x) = -T \times \log\left[\sum_{y \in \mathcal{Y}}\exp\left(\frac{p(y \mid x)}{T}\right)\right]$
$score_{MD}(x, D_{tr}) = \sqrt{(x - \mu)^\top\Sigma^{-1}(x - \mu)}$

```python
OOD-Sentiment-LLM/
├─ ood_scoring/
│  ├─ __init__.py
│  ├─ scoring.py
│  └─ examples/
│     └─ demo_fake_data.py   # 윤혁 test file
```
```python
# test code
# python -m ood_scoring.examples.demo_fake_data

from ood_scoring import (
    score_msp,
    score_energy_from_probs,
    fit_md,
    score_md
)

msp = score_msp(probs)
energy = score_energy_from_probs(probs)
mu, inv_cov = fit_md(train_feats)
md_scores = score_md(test_feats, mu, inv_cov)
```

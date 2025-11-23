import os
import json
import ujson
import numpy as np
from tqdm import tqdm
from typing import Dict, List
from pathlib import Path

import tiktoken
from openai import OpenAI

# 1) 경로/설정 
THIS_FILE   = Path(__file__).resolve()
SRC_DIR     = THIS_FILE.parents[1]                 # .../OOD-Sentiment-LLM/src
DATASET_DIR = SRC_DIR / "dataset"
ARTIFACT_DIR = SRC_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILES = [
    str(DATASET_DIR / "All_Beauty.jsonl_25k_test.jsonl"),
    str(DATASET_DIR / "Baby_Products.jsonl_25k_test.jsonl"),
    str(DATASET_DIR / "Grocery_and_Gourmet_Food.jsonl_25k_test.jsonl"),
    str(DATASET_DIR / "Industrial_and_Scientific.jsonl_25k_test.jsonl"),
]

# OpenAI 세팅
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise RuntimeError("환경변수 OPENAI_API_KEY가 설정되지 않았습니다.")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "text-embedding-ada-002")

# 토큰 길이 제한(필요 시 조정)
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))

# 2) 데이터 로드 (jsonl) 
#   - 이미 json에 group 필드가 N/P/OOD로 라벨링되어 있으므로 그대로 사용
def load_jsonl_files(paths: List[str]) -> List[Dict]:
    data = []
    for p in paths:
        pth = Path(p)
        if not pth.exists():
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {p}")
        with open(pth, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = ujson.loads(line)
                except ValueError:
                    continue  # JSON 파싱 실패 행 스킵
                # text, group 필드 필요
                if "text" in obj and "group" in obj:
                    # group은 'N' / 'P' / 'OOD' 가정
                    g = obj["group"]
                    if g in {"N", "P", "OOD"}:
                        data.append({"text": obj["text"], "group": g})
    return data

# 3) GPT 토크나이저(tiktoken)로 길이 자르기 
_enc = tiktoken.get_encoding("cl100k_base")

def truncate_by_tokens(text: str, max_tokens: int = MAX_TOKENS) -> str:
    toks = _enc.encode(text)
    if len(toks) <= max_tokens:
        return text
    return _enc.decode(toks[:max_tokens])

# 4) OpenAI Embeddings 헬퍼 
_client = OpenAI(api_key=OPENAI_API_KEY)

def embed_texts(texts: List[str], model: str = EMBED_MODEL_NAME) -> np.ndarray:
    # 빈문자 방지
    safe_texts = [t if (t and t.strip()) else " " for t in texts]
    resp = _client.embeddings.create(model=model, input=safe_texts)
    embs = [d.embedding for d in resp.data]
    return np.asarray(embs, dtype=np.float32)  # (B, 1536)

# 5) 메인 파이프라인 
def main():
    print("[1] 데이터 로드...")
    raw = load_jsonl_files(DATA_FILES)
    print(f"총 로우 수: {len(raw)}")

    # 텍스트 전처리(토큰 트렁케이션) + group 그대로 사용
    cleaned = []
    for r in raw:
        txt = truncate_by_tokens(r["text"], MAX_TOKENS)
        cleaned.append({"text": txt, "group": r["group"]})

    # 분리: PN(테스트용 감성라벨) / OOD(별도 저장)
    pn_records  = [x for x in cleaned if x["group"] in {"N", "P"}]
    ood_records = [x for x in cleaned if x["group"] == "OOD"]

    print(f"PN(=N/P): {len(pn_records)} | OOD: {len(ood_records)}")

    # ---------------- PN 임베딩 & 라벨(0/1) ----------------
    print("[2] PN 임베딩 추출 (OpenAI Embeddings)...")
    pn_texts = [rec["text"] for rec in pn_records]
    # 라벨: N->0, P->1
    pn_y = np.asarray([0 if rec["group"] == "N" else 1 for rec in pn_records], dtype=np.int64)

    pn_emb_list = []
    for i in tqdm(range(0, len(pn_texts), BATCH_SIZE)):
        chunk = pn_texts[i:i+BATCH_SIZE]
        embs = embed_texts(chunk, EMBED_MODEL_NAME)     # (B, 1536)
        pn_emb_list.append(embs)
    pn_X = np.vstack(pn_emb_list) if pn_emb_list else np.zeros((0, 1536), dtype=np.float32)

    # ---------------- OOD 임베딩 ----------------
    print("[3] OOD 임베딩 추출 (OpenAI Embeddings)...")
    ood_texts = [rec["text"] for rec in ood_records]
    ood_emb_list = []
    for i in tqdm(range(0, len(ood_texts), BATCH_SIZE)):
        chunk = ood_texts[i:i+BATCH_SIZE]
        embs = embed_texts(chunk, EMBED_MODEL_NAME)
        ood_emb_list.append(embs)
    ood_X = np.vstack(ood_emb_list) if ood_emb_list else np.zeros((0, 1536), dtype=np.float32)

    # ---------------- 저장 ----------------
    print("[4] 저장...")
    # 테스트 평가는 PN(=N/P)을 대상으로 하므로 X/Y 파일명에 저장
    np.save(ARTIFACT_DIR / "embeddings_test_X.npy", pn_X)
    np.save(ARTIFACT_DIR / "embeddings_test_Y.npy", pn_y)     # ← 대문자 Y

    # OOD는 별도(후속 OOD 플롯/스코어용)
    np.save(ARTIFACT_DIR / "embeddings_ood_test_X.npy", ood_X)

    print("완료")

if __name__ == "__main__":
    main()
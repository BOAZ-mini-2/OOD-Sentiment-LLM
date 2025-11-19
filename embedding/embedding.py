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
# 파일 위치를 실행 위치와 무관하게 안전하게 처리
THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parents[1]                   # .../OOD-Sentiment-LLM/src
DATASET_DIR = SRC_DIR / "dataset"
ARTIFACT_DIR = SRC_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILES = [
    str(DATASET_DIR / "All_Beauty.jsonl_25k.jsonl"),
    str(DATASET_DIR / "Baby_Products.jsonl_25k.jsonl"),
    str(DATASET_DIR / "Grocery_and_Gourmet_Food.jsonl_25k.jsonl"),
    str(DATASET_DIR / "Industrial_and_Scientific.jsonl_25k.jsonl"),
]

# OpenAI 세팅
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise RuntimeError("환경변수 OPENAI_API_KEY가 설정되지 않았습니다.")

# 임베딩 모델 (요구사항)
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "text-embedding-ada-002")

# 토큰 관련 설정
# - text-embedding-ada-002는 cl100k_base 토크나이저를 사용
# - 안전한 학습·속도를 위해 적당한 최대 토큰 길이로 자름(필요시 조정)
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))

# 배치 크기(한 번에 임베딩 API로 보낼 문장 수)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))

# 2) 라벨 재정의
#   (0,1)->neg, (3,4)->pos, (2)->ood, 그 외 skip)
def remap_rating_to_label(rating: int) -> str:
    if rating in {0, 1}:
        return "neg"
    if rating == 2:
        return "ood"
    if rating in {3, 4}:
        return "pos"
    return "skip"


# 3) 데이터 로드 (jsonl)
def load_jsonl_files(paths: List[str]) -> List[Dict]:
    data = []
    for p in paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {p}")
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = ujson.loads(line)
                except ValueError:
                    continue  # JSON 파싱 실패 행 스킵
                if "rating" in obj and "text" in obj:
                    data.append({"rating": obj["rating"], "text": obj["text"]})
    return data


# 4) GPT 토크나이저(tiktoken)로 토큰 길이 자르기
#    - 임베딩 API는 문자열 입력이므로, 토큰 기준으로 자른 뒤 다시 문자열로 복원
_enc = tiktoken.get_encoding("cl100k_base")

def truncate_by_tokens(text: str, max_tokens: int = MAX_TOKENS) -> str:
    """
    text → tokens → 앞에서 max_tokens개만 → 다시 text
    주의: 토큰 기반 복원이라 공백 등 세부가 미세하게 달라질 수 있으나,
         임베딩 품질/일관성에는 문제 없음.
    """
    toks = _enc.encode(text)
    if len(toks) <= max_tokens:
        return text
    toks = toks[:max_tokens]
    return _enc.decode(toks)


# 5) OpenAI Embeddings 헬퍼
_client = OpenAI(api_key=OPENAI_API_KEY)

def embed_texts(texts: List[str], model: str = EMBED_MODEL_NAME) -> np.ndarray:
    """
    texts: 문자열 리스트 (배치)
    반환: (len(texts), 1536) 형태의 float32 numpy 배열 (ada-002는 1536차원)
    """
    # 공백/빈 문자열 방지
    safe_texts = [t if (t and t.strip()) else " " for t in texts]
    resp = _client.embeddings.create(model=model, input=safe_texts)
    # 응답 순서가 입력 순서와 동일하므로 그대로 추출
    embs = [d.embedding for d in resp.data]
    return np.asarray(embs, dtype=np.float32)


# 6) 메인 파이프라인
def main():
    print("[1] 데이터 로드...")
    raw = load_jsonl_files(DATA_FILES)
    print(f"총 로우 수: {len(raw)}")

    # 라벨 재정의 & 텍스트 전처리(토큰 트렁케이션)
    mapped = []
    for r in raw:
        tag = remap_rating_to_label(r["rating"])
        if tag == "skip":
            continue
        # GPT 토크나이저 기준 토큰 길이 제한
        txt = truncate_by_tokens(r["text"], MAX_TOKENS)
        mapped.append({"text": txt, "label_str": tag})

    # 데이터 분리
    train_records = [x for x in mapped if x["label_str"] in {"neg", "pos"}]
    ood_records   = [x for x in mapped if x["label_str"] == "ood"]

    print(f"Train(PN): {len(train_records)} | OOD(2): {len(ood_records)}")

    # ============ 임베딩 추출: Train(PN) ============
    print("[2] Train 임베딩 추출 (OpenAI Embeddings)...")
    train_texts, train_y = [], []
    for rec in train_records:
        train_texts.append(rec["text"])
        # neg→0, pos→1
        train_y.append(0 if rec["label_str"] == "neg" else 1)
    train_y = np.asarray(train_y, dtype=np.int64)

    # 배치로 쪼개어 API 호출
    train_emb_list = []
    for i in tqdm(range(0, len(train_texts), BATCH_SIZE)):
        chunk = train_texts[i:i+BATCH_SIZE]
        embs = embed_texts(chunk, EMBED_MODEL_NAME)     # (B, 1536)
        train_emb_list.append(embs)
    train_X = np.vstack(train_emb_list) if train_emb_list else np.zeros((0, 1536), dtype=np.float32)

    # ============ 임베딩 추출: OOD(=라벨 2) ============
    print("[3] OOD 임베딩 추출 (OpenAI Embeddings)...")
    ood_texts = [rec["text"] for rec in ood_records]
    ood_emb_list = []
    for i in tqdm(range(0, len(ood_texts), BATCH_SIZE)):
        chunk = ood_texts[i:i+BATCH_SIZE]
        embs = embed_texts(chunk, EMBED_MODEL_NAME)
        ood_emb_list.append(embs)
    ood_X = np.vstack(ood_emb_list) if ood_emb_list else np.zeros((0, 1536), dtype=np.float32)

    # ============ 저장 ============
    print("[4] 저장...")
    np.save(ARTIFACT_DIR / "embeddings_train_X.npy", train_X)
    np.save(ARTIFACT_DIR / "embeddings_train_y.npy", train_y)
    np.save(ARTIFACT_DIR / "embeddings_ood_X.npy",   ood_X)

    # 텍스트 백업(jsonl)
    with open(ARTIFACT_DIR / "train_texts.jsonl", "w", encoding="utf-8") as f:
        for t, y in zip(train_texts, train_y.tolist()):
            f.write(json.dumps({"text": t, "label": int(y)}, ensure_ascii=False) + "\n")

    with open(ARTIFACT_DIR / "ood_texts.jsonl", "w", encoding="utf-8") as f:
        for t in ood_texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    print("완료")


if __name__ == "__main__":
    main()

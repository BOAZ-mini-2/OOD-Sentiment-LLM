# 👉 각자 환경에 맞게 임베딩 데이터 루트 경로를 수정해 주세요.

예) scatter.py / plot.py 초반부:

**임베딩 데이터 루트(필요 시 본인 환경에 맞게 수정)**

embedding/ 폴더로 저장하는 경우

`CANDIDATES = [Path("embedding"), Path("src/artifacts"), Path("artifacts")]`

또는 고정 경로로:

`ARTIFACT_DIR = Path("embedding").resolve()  # 본인 폴더명으로 변경`


(필수 파일: embeddings_train_X.npy, embeddings_train_y.npy, embeddings_ood_X.npy)

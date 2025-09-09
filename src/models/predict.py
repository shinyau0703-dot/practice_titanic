# src/models/predict.py

"""

!!執行方法!!

從專案根目錄執行
確保你現在在 D:\practive_w1（專案根目錄），再用 模組方式執行：

cd D:\practive_w1
python -m src.models.predict

這樣 Python 會把 D:\practive_w1 當成 package root，src/... 就能找到

"""

"""
Titanic 推論腳本
- 載入 models_store/model.joblib（內含前處理 + 模型的 Pipeline）
- 讀取 data/processed/test_clean.csv
- 產生 submission.csv（PassengerId, Survived）

對外：
  run_predict(processed_dir, model_dir, out_csv)
或直接命令列：
  python src/models/predict.py --processed-dir data/processed --model-dir models_store --out experiments/submission.csv
"""

from __future__ import annotations
from pathlib import Path
import argparse
import logging

import pandas as pd
from joblib import load

from src.models.train import get_logger

logger = get_logger("predict")


def run_predict(processed_dir: Path, model_dir: Path, out_csv: Path) -> None:
    """
    載入已訓練的 pipeline，對 test_clean.csv 做預測並輸出 submission.csv
    """
    processed_dir = Path(processed_dir)
    model_dir = Path(model_dir)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    test_path = processed_dir / "test_clean.csv"
    model_path = model_dir / "model.joblib"

    if not test_path.exists():
        raise FileNotFoundError(f"找不到測試資料：{test_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"找不到已存模型：{model_path}")

    logger.info(f"讀取測試資料：{test_path}")
    test_df = pd.read_csv(test_path)

    # 保存 PassengerId 供輸出
    if "PassengerId" not in test_df.columns:
        raise ValueError("test_clean.csv 缺少 PassengerId 欄位")
    pid = test_df["PassengerId"].values

    # 與訓練一致：移除非特徵欄（若存在）
    X_test = test_df.drop(columns=["PassengerId", "Survived"], errors="ignore")

    logger.info(f"載入模型：{model_path}")
    clf = load(model_path)

    logger.info("進行推論...")
    y_pred = clf.predict(X_test)

    sub = pd.DataFrame({"PassengerId": pid, "Survived": y_pred.astype(int)})
    sub.to_csv(out_csv, index=False)
    logger.info(f"已輸出提交檔：{out_csv}")


# --------------- CLI ---------------
def parse_args():
    p = argparse.ArgumentParser(description="Predict Titanic test set and export submission.csv")
    p.add_argument("--processed-dir", type=str, default="data/processed")
    p.add_argument("--model-dir", type=str, default="models_store")
    p.add_argument("--out", type=str, default="experiments/submission.csv")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    run_predict(Path(args.processed_dir), Path(args.model_dir), Path(args.out))
    logger.info("Done.")


if __name__ == "__main__":
    main()

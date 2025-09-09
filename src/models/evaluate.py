# src/models/evaluate.py


"""

!!執行方法!!  

專案根目錄 + 模組方式（推薦）
確保你在專案根目錄 (D:\practive_w1)，用 -m 執行：

cd D:\practive_w1
python -m src.models.evaluate

這樣 Python 會把 D:\practive_w1 當成 package root，src/... 的 import 才能正確。

"""




"""
Titanic 評估腳本
- 預設使用 K-fold 交叉驗證（不依賴已存模型），輸出泛化評估
- 可選 --use-saved-model：載入 models_store/model.joblib，計算「訓練集」上的分數（僅供對照）

輸出：
  experiments/eval_report.txt
  experiments/eval_report.json
"""

from __future__ import annotations
from pathlib import Path
import argparse
import logging
import json

import pandas as pd
from joblib import load
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 從 train.py 共用前處理器（避免重複）
from src.models.train import build_preprocessor, get_logger, RANDOM_STATE

logger = get_logger("evaluate")


def _eval_with_saved_model(processed_dir: Path, model_dir: Path) -> dict:
    """
    載入已訓練好的 pipeline (model.joblib)，在「整個 train_clean.csv」上評估。
    注意：這是訓練集上的分數，僅供 sanity check，不代表泛化能力。
    """
    data_path = processed_dir / "train_clean.csv"
    model_path = model_dir / "model.joblib"
    if not data_path.exists():
        raise FileNotFoundError(f"找不到處理後資料：{data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"找不到已存模型：{model_path}")

    logger.info(f"讀取資料：{data_path}")
    df = pd.read_csv(data_path)
    if "Survived" not in df.columns:
        raise ValueError("train_clean.csv 缺少目標欄位 'Survived'")

    y = df["Survived"].astype(int)
    X = df.drop(columns=["Survived", "PassengerId"], errors="ignore")

    logger.info(f"載入模型：{model_path}")
    clf = load(model_path)

    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    cls_rep = classification_report(y, y_pred, digits=4)
    cm = confusion_matrix(y, y_pred)

    logger.info(f"[UseSavedModel] Train Accuracy: {acc:.4f}")
    logger.info("Classification Report:\n" + cls_rep)
    logger.info(f"Confusion Matrix:\n{cm}")

    return {
        "mode": "use_saved_model_train_score",
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": cls_rep,
    }


def _eval_with_cv(processed_dir: Path, n_splits: int = 5) -> dict:
    """
    使用 KFold 交叉驗證（預設 5-fold）評估 baseline：
      Pipeline = build_preprocessor() + LogisticRegression
    注意：這會重新訓練（僅為評估），不覆蓋已存模型。
    """
    data_path = processed_dir / "train_clean.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"找不到處理後資料：{data_path}")

    logger.info(f"讀取資料：{data_path}")
    df = pd.read_csv(data_path)
    if "Survived" not in df.columns:
        raise ValueError("train_clean.csv 缺少目標欄位 'Survived'")

    y = df["Survived"].astype(int)
    X = df.drop(columns=["Survived", "PassengerId"], errors="ignore")

    pipe = Pipeline(
        steps=[
            ("pre", build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    logger.info(f"開始 {n_splits}-fold 交叉驗證...")
    scores = cross_val_score(pipe, X, y, cv=kf, scoring="accuracy", n_jobs=None)
    logger.info(f"CV Accuracies: {', '.join(f'{s:.4f}' for s in scores)}")
    logger.info(f"CV Mean: {scores.mean():.4f}  |  Std: {scores.std():.4f}")

    return {
        "mode": f"{n_splits}fold_cv",
        "cv_scores": [float(s) for s in scores],
        "cv_mean": float(scores.mean()),
        "cv_std": float(scores.std()),
    }


def run_eval(processed_dir: Path, model_dir: Path, exp_dir: Path, use_saved_model: bool = False, n_splits: int = 5) -> dict:
    """
    高階評估入口：
      - use_saved_model=True：用既有模型在 train_clean 上做訓練分數
      - 否則：做 KFold 交叉驗證
    """
    processed_dir = Path(processed_dir)
    model_dir = Path(model_dir)
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    if use_saved_model:
        report = _eval_with_saved_model(processed_dir, model_dir)
    else:
        report = _eval_with_cv(processed_dir, n_splits=n_splits)

    # 輸出報告
    out_txt = exp_dir / "eval_report.txt"
    out_json = exp_dir / "eval_report.json"
    with open(out_txt, "w", encoding="utf-8") as f:
        if report["mode"] == "use_saved_model_train_score":
            f.write(f"Mode: {report['mode']}\n")
            f.write(f"Train Accuracy: {report['accuracy']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report["classification_report"] + "\n")
            f.write("Confusion Matrix:\n")
            f.write(str(report["confusion_matrix"]) + "\n")
        else:
            f.write(f"Mode: {report['mode']}\n")
            f.write("CV Accuracies:\n")
            f.write(", ".join(f"{s:.4f}" for s in report["cv_scores"]) + "\n")
            f.write(f"CV Mean: {report['cv_mean']:.4f}\n")
            f.write(f"CV Std : {report['cv_std']:.4f}\n")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"已輸出評估報告：{out_txt}")
    return report


# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Titanic model")
    p.add_argument("--processed-dir", type=str, default="data/processed")
    p.add_argument("--model-dir", type=str, default="models_store")
    p.add_argument("--exp-dir", type=str, default="experiments")
    p.add_argument("--use-saved-model", action="store_true", help="Load models_store/model.joblib and score on full train")
    p.add_argument("--cv", type=int, default=5, help="K-fold for cross-validation (ignored if --use-saved-model)")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    report = run_eval(Path(args.processed_dir), Path(args.model_dir), Path(args.exp_dir),
                      use_saved_model=args.use_saved_model, n_splits=args.cv)
    logger.info(f"Done. mode={report['mode']}")


if __name__ == "__main__":
    main()

# src/models/train.py
"""
Titanic baseline 訓練腳本（含 5-fold CV）
- 讀取 data/processed/train_clean.csv
- 5-fold StratifiedKFold：回傳每折 acc / f1，平均值
- 前處理：OneHot(Sex, Embarked) + StandardScaler(數值)
- 模型：LogisticRegression
- 輸出：
    models_store/model.joblib            （用全資料再訓練的整條 Pipeline）
    experiments/baseline_results.txt     （Accuracy / F1 / 報表概要）
    experiments/baseline_results.json    （機器可讀的 metrics）
- 提供 train_and_eval() 讓 app.py 呼叫，回傳 dict：
    {
      "model_family": "logreg_v1",
      "seed": 42,
      "fold_accs": [...],
      "fold_f1s":  [...],
      "avg_acc": 0.8123,
      "avg_f1":  0.7891
    }
"""

from __future__ import annotations
from pathlib import Path
import argparse
import logging
import json
from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# --------------------
# 基本設定
# --------------------
RANDOM_STATE = 42
MODEL_FAMILY = "logreg_v1"  # 寫進 DB 的模型名

def get_logger(name: str = "train"):
    """統一 logger（若你的專案已有 utils.get_logger，可改為匯入使用）"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(h)
    return logger

logger = get_logger()

# --------------------
# 前處理器
# --------------------
def build_preprocessor() -> ColumnTransformer:
    """
    建立 ColumnTransformer：
      - 類別：Sex, Embarked → OneHot(handle_unknown='ignore')
      - 數值：Age, Fare, FamilySize, Pclass, SibSp, Parch → StandardScaler
    備註：Pclass/SibSp/Parch 雖為整數類別，但對 LogReg 先當連續數值做標準化可行；
         若改 tree/boosting，可調整這組欄位。
    """
    cat_features = ["Sex", "Embarked"]
    num_features = ["Age", "Fare", "FamilySize", "Pclass", "SibSp", "Parch"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", StandardScaler(), num_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor

def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("pre", build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE)),
        ]
    )

# --------------------
# 5-fold 訓練主流程（提供給 app.py）
# --------------------
def train_and_eval(
    processed_dir: str | Path = "data/processed",
    model_dir: str | Path = "models_store",
    exp_dir: str | Path = "experiments",
    seed: int = RANDOM_STATE,
    n_splits: int = 5,
) -> Dict:
    """
    讀 processed/train_clean.csv → 5-fold CV → 回傳每折 acc/f1 與平均
    並將全資料重訓的 model.joblib 存到 models_store
    """
    processed_dir = Path(processed_dir)
    model_dir = Path(model_dir)
    exp_dir = Path(exp_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    data_path = processed_dir / "train_clean.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"找不到處理後資料：{data_path}")

    logger.info(f"讀取資料：{data_path}")
    df = pd.read_csv(data_path)

    if "Survived" not in df.columns:
        raise ValueError("train_clean.csv 缺少目標欄位 'Survived'")

    y = df["Survived"].astype(int)
    X = df.drop(columns=["Survived", "PassengerId"], errors="ignore")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_accs: List[float] = []
    fold_f1s: List[float] = []
    reports: List[str] = []
    cms: List[np.ndarray] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        clf = build_pipeline()
        logger.info(f"[Fold {fold}] 訓練中... train={X_tr.shape}, valid={X_va.shape}")
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_va)
        acc = accuracy_score(y_va, y_pred)
        f1 = f1_score(y_va, y_pred)

        fold_accs.append(float(acc))
        fold_f1s.append(float(f1))

        rep = classification_report(y_va, y_pred, digits=4)
        cm = confusion_matrix(y_va, y_pred)
        reports.append(rep)
        cms.append(cm)

        logger.info(f"[Fold {fold}] acc={acc:.4f}, f1={f1:.4f}")

    avg_acc = float(np.mean(fold_accs)) if fold_accs else None
    avg_f1  = float(np.mean(fold_f1s)) if fold_f1s else None
    logger.info(f"CV 平均：acc={avg_acc:.4f}, f1={avg_f1:.4f}")

    # 用全資料重訓並存檔（供預測用）
    final_clf = build_pipeline()
    final_clf.fit(X, y)
    model_path = model_dir / "model.joblib"
    dump(final_clf, model_path)
    logger.info(f"已保存模型：{model_path}")

    # 寫出文字與 JSON 結果
    results_txt = exp_dir / "baseline_results.txt"
    with open(results_txt, "w", encoding="utf-8") as f:
        f.write(f"Model: {MODEL_FAMILY}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"CV folds: {n_splits}\n")
        f.write(f"Mean Accuracy: {avg_acc:.4f}\n")
        f.write(f"Mean F1: {avg_f1:.4f}\n\n")
        for i, (rep, cm) in enumerate(zip(reports, cms)):
            f.write(f"--- Fold {i} ---\n")
            f.write(rep + "\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + "\n\n")
    logger.info(f"已輸出評估結果：{results_txt}")

    results_json = exp_dir / "baseline_results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_family": MODEL_FAMILY,
                "seed": int(seed),
                "n_splits": int(n_splits),
                "fold_accs": [float(x) for x in fold_accs],
                "fold_f1s": [float(x) for x in fold_f1s],
                "avg_acc": avg_acc,
                "avg_f1": avg_f1,
                "model_path": str(model_path),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 回傳給 app.py 寫 DB 用
    return {
        "model_family": MODEL_FAMILY,
        "seed": int(seed),
        "fold_accs": [float(x) for x in fold_accs],
        "fold_f1s": [float(x) for x in fold_f1s],
        "avg_acc": avg_acc,
        "avg_f1": avg_f1,
    }

# --------------------
# 舊版單一切分流程（保留 CLI 相容）
# --------------------
def run_train(
    processed_dir: Path | str,
    model_dir: Path | str,
    exp_dir: Path | str,
) -> dict:
    """保留單一切分版本（與你原本相容）；建議還是用 train_and_eval()"""
    return train_and_eval(processed_dir, model_dir, exp_dir)

# --------------------
# CLI 入口（可單獨跑本檔）
# --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train Titanic baseline with 5-fold CV")
    p.add_argument("--processed-dir", type=str, default="data/processed", help="Directory of processed CSVs")
    p.add_argument("--model-dir", type=str, default="models_store", help="Where to save model.joblib")
    p.add_argument("--exp-dir", type=str, default="experiments", help="Where to save metrics/results")
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--folds", type=int, default=5)
    return p.parse_args()

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    metrics = train_and_eval(
        processed_dir=args.processed_dir,
        model_dir=args.model_dir,
        exp_dir=args.exp_dir,
        seed=args.seed,
        n_splits=args.folds,
    )
    logger.info(f"Done. metrics={metrics}")

if __name__ == "__main__":
    main()

# src/models/train.py
"""
Titanic baseline 訓練腳本
- 讀取 data/processed/train_clean.csv
- 切分 train/valid (80/20, stratify=Survived)
- 前處理：OneHot(Sex, Embarked) + StandardScaler(數值)
- 模型：LogisticRegression
- 輸出：
    models_store/model.joblib    （整條 Pipeline）
    experiments/baseline_results.txt（Accuracy / 分類報表 / 混淆矩陣）
- 對外介面：
    run_train(processed_dir, model_dir, exp_dir) -> dict
    也可直接 CLI 執行：python src/models/train.py --processed-dir ... --model-dir ... --exp-dir ...
"""

from __future__ import annotations
from pathlib import Path
import argparse
import logging
import json

import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --------------------
# 基本設定
# --------------------
RANDOM_STATE = 42

def get_logger(name: str = "train"):
    """統一 logger（若你的專案已有 utils.logging.get_logger，可改為匯入使用）"""
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

# --------------------
# 主訓練流程（給 app.py 呼叫）
# --------------------
def run_train(processed_dir: Path, model_dir: Path, exp_dir: Path) -> dict:
    """
    讀 processed/train_clean.csv → 切分 → 建立 pipeline(前處理+模型) → 訓練 → 評估 → 輸出
    回傳：metrics dict（含 accuracy）
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

    # 目標與特徵
    if "Survived" not in df.columns:
        raise ValueError("train_clean.csv 缺少目標欄位 'Survived'")
    y = df["Survived"].astype(int)
    X = df.drop(columns=["Survived", "PassengerId"], errors="ignore")

    # 切分資料（Stratify 保持目標比例）
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"切分完成：Train={X_train.shape}, Valid={X_valid.shape}")

    # 建立 Pipeline：前處理 + 模型
    clf = Pipeline(
        steps=[
            ("pre", build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000, n_jobs=None, random_state=RANDOM_STATE)),
        ]
    )

    # 訓練
    logger.info("開始訓練 LogisticRegression baseline...")
    clf.fit(X_train, y_train)

    # 驗證評估
    y_pred = clf.predict(X_valid)
    acc = accuracy_score(y_valid, y_pred)
    cls_rep = classification_report(y_valid, y_pred, digits=4)
    cm = confusion_matrix(y_valid, y_pred)

    logger.info(f"Validation Accuracy: {acc:.4f}")
    logger.info("Classification Report:\n" + cls_rep)
    logger.info(f"Confusion Matrix:\n{cm}")

    # 輸出模型（整條 pipeline）
    model_path = model_dir / "model.joblib"
    dump(clf, model_path)
    logger.info(f"已保存模型：{model_path}")

    # 輸出結果到 experiments
    results_txt = exp_dir / "baseline_results.txt"
    with open(results_txt, "w", encoding="utf-8") as f:
        f.write(f"Validation Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(cls_rep + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")
    logger.info(f"已輸出評估結果：{results_txt}")

    # 也輸出一份 JSON 方便程式讀取（可選）
    results_json = exp_dir / "baseline_results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": float(acc),
                "confusion_matrix": cm.tolist(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    metrics = {"accuracy": float(acc), "model_path": str(model_path)}
    return metrics

# --------------------
# CLI 入口
# --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train Titanic baseline model")
    p.add_argument("--processed-dir", type=str, default="data/processed", help="Directory of processed CSVs")
    p.add_argument("--model-dir", type=str, default="models_store", help="Where to save model.joblib")
    p.add_argument("--exp-dir", type=str, default="experiments", help="Where to save metrics/results")
    return p.parse_args()

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    metrics = run_train(Path(args.processed_dir), Path(args.model_dir), Path(args.exp_dir))
    logger.info(f"Done. metrics={metrics}")

if __name__ == "__main__":
    main()

# src/models/train_kfold.py
import argparse
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import hashlib

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import mlflow
    import mlflow.sklearn as mlflow_sklearn
    MLFLOW_OK = True
except Exception:
    mlflow = None
    mlflow_sklearn = None
    MLFLOW_OK = False


# 產生檔案的 SHA256 指紋，方便重現性比對
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# 取得目前 git commit 短雜湊，寫入實驗資訊
def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# 建立實驗輸出資料夾（以時間戳命名）並回傳路徑
def make_run_dir(root="experiments") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = Path(root) / ts
    p.mkdir(parents=True, exist_ok=True)
    return p


# 建立檔案與主控台雙輸出的 logger，回傳 logger 與 log 檔路徑
def setup_logger(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_kfold_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger("kfold")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger, log_path


# 建立前處理器：數值欄位做補值與標準化，類別欄位做補值與 One-Hot
def build_preprocessor(num_cols, cat_cols) -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])
    try:
        categorical_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
    except TypeError:
        categorical_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )


# 建立整體 Pipeline：前處理 + LogisticRegression（設定隨機種子可重現）
def build_pipeline(preprocessor, seed: int) -> Pipeline:
    return Pipeline(steps=[
        ("pre", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=seed, n_jobs=1, solver="lbfgs")),
    ])


# 解析參數、執行 Stratified 5-fold 訓練、記錄每折結果與平均、輸出模型與實驗資訊
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="processed CSV 路徑（含目標欄）")
    ap.add_argument("--target", required=True, help="目標欄位名稱（例如 Survived）")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42, help="CV 與模型隨機種子")
    args = ap.parse_args()

    data_path = Path(args.data)
    assert data_path.exists(), f"找不到資料：{data_path}"
    df = pd.read_csv(data_path)

    y = df[args.target].values
    X = df.drop(columns=[args.target])

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    np.random.seed(args.seed)

    models_dir = Path("models"); models_dir.mkdir(exist_ok=True)
    run_dir = make_run_dir("experiments")
    logs_dir = Path("logs")
    logger, log_path = setup_logger(logs_dir)

    logger.info(f"Start k-fold training | data={data_path} | target={args.target} | folds={args.folds} | seed={args.seed}")
    logger.info(f"Columns: num={num_cols} | cat={cat_cols}")

    preprocessor = build_preprocessor(num_cols, cat_cols)
    kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    if MLFLOW_OK:
        tracking_dir = Path("experiments") / "mlruns"
        mlflow.set_tracking_uri("file:" + str(tracking_dir.resolve()))
        mlflow.set_experiment("Titanic_5fold")
        try:
            mlflow_sklearn.autolog(log_input_examples=False, log_models=False)
        except Exception:
            pass

    fold_summ = []

    parent_cm = mlflow.start_run(run_name=f"kfold_seed{args.seed}_f{args.folds}") if MLFLOW_OK else nullcontext()
    with parent_cm:
        if MLFLOW_OK:
            mlflow.log_params({
                "cv_type": "StratifiedKFold",
                "cv_folds": args.folds,
                "random_seed": args.seed,
                "model_family": "logreg",
                "n_num_cols": len(num_cols),
                "n_cat_cols": len(cat_cols),
            })
            mlflow.set_tags({
                "git_commit": git_commit(),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "data_path": str(data_path),
                "data_sha256": sha256_file(data_path),
            })

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            logger.info(f"[Fold {fold}] n_train={len(tr_idx)} n_valid={len(va_idx)}")

            pipe = build_pipeline(preprocessor, args.seed)
            pipe.fit(X_tr, y_tr)
            logger.info(f"[Fold {fold}] fitted; evaluating…")

            pred = pipe.predict(X_va)
            acc = accuracy_score(y_va, pred)
            f1 = f1_score(y_va, pred)
            logger.info(f"[Fold {fold}] acc={acc:.4f} f1={f1:.4f}")

            model = pipe.named_steps["model"]
            params = model.get_params(deep=False)

            fold_info = {
                "fold": fold,
                "metrics": {"acc": acc, "f1": f1},
                "params": params,
                "n_train": int(len(tr_idx)),
                "n_valid": int(len(va_idx)),
                "num_cols": num_cols,
                "cat_cols": cat_cols,
            }
            fold_summ.append(fold_info)

            model_path = models_dir / f"fold_{fold}.joblib"
            joblib.dump(pipe, model_path)
            (run_dir / f"fold_{fold}.json").write_text(
                json.dumps(fold_info, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            child_cm = mlflow.start_run(run_name=f"fold_{fold}", nested=True) if MLFLOW_OK else nullcontext()
            with child_cm:
                if MLFLOW_OK:
                    mlflow.log_params(params)
                    mlflow.log_metrics({"acc": acc, "f1": f1})
                    mlflow.log_artifact(str(model_path))
                    mlflow.log_artifact(str(run_dir / f"fold_{fold}.json"))

        avg_acc = float(np.mean([f["metrics"]["acc"] for f in fold_summ]))
        avg_f1 = float(np.mean([f["metrics"]["f1"] for f in fold_summ]))
        logger.info(f"[AVG] acc={avg_acc:.4f} f1={avg_f1:.4f}")

        RUN_INFO = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "git_commit": git_commit(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": {
                "numpy": __import__("numpy").__version__,
                "pandas": __import__("pandas").__version__,
                "scikit_learn": __import__("sklearn").__version__,
                "joblib": __import__("joblib").__version__,
            },
            "data": {
                "path": str(data_path),
                "sha256": sha256_file(data_path),
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
                "target": args.target,
            },
            "cv": {"folds": args.folds, "shuffle": True, "seed": args.seed, "type": "StratifiedKFold"},
            "model_family": "logreg",
            "folds": fold_summ,
            "metrics_avg": {"acc": avg_acc, "f1": avg_f1},
        }
        run_info_path = run_dir / "RUN_INFO.json"
        run_info_path.write_text(json.dumps(RUN_INFO, indent=2, ensure_ascii=False), encoding="utf-8")

        if MLFLOW_OK:
            mlflow.log_metrics({"avg_acc": avg_acc, "avg_f1": avg_f1})
            mlflow.log_artifact(str(run_info_path))
            mlflow.log_artifacts(str(run_dir))
            mlflow.log_artifact(str(log_path))

    print(f"[OK] 5-fold 完成。平均 acc={avg_acc:.4f}, f1={avg_f1:.4f}")
    print(f"模型輸出：{models_dir.resolve()}")
    print(f"實驗輸出：{run_dir.resolve()}（MLflow 路徑：experiments/mlruns）")


# 提供「空的 with 區塊」語法，讓未安裝 MLflow 時也能用同樣程式流程
class nullcontext:
    def __enter__(self): return None
    def __exit__(self, *exc): return False


if __name__ == "__main__":
    main()

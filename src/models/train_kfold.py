# src/models/train_kfold.py
import argparse, json, os, hashlib, platform, subprocess, sys
from datetime import datetime
from pathlib import Path

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

import mlflow  # ← 新增

# ---------- 小工具 ----------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"

def make_run_dir(root="experiments") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = Path(root) / ts
    p.mkdir(parents=True, exist_ok=True)
    return p

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

    # 欄位型別
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # 固定隨機性
    np.random.seed(args.seed)

    # 目錄
    models_dir = Path("models"); models_dir.mkdir(exist_ok=True)
    run_dir = make_run_dir("experiments")

    # 前處理
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])
    # sklearn 1.2+ 用 sparse_output，舊版用 sparse；為了相容，這樣處理：
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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )

    def build_pipeline(seed: int):
        return Pipeline(steps=[
            ("pre", preprocessor),
            ("model", LogisticRegression(max_iter=1000, random_state=seed, n_jobs=1, solver="lbfgs"))
        ])

    kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    # ===== MLflow 設定 =====
    mlflow.set_tracking_uri("file:" + str((Path("experiments") / "mlruns").resolve()))
    mlflow.set_experiment("Titanic_5fold")

    fold_summ = []

    # 父 run（聚合本次 5 折）
    with mlflow.start_run(run_name=f"kfold_seed{args.seed}_f{args.folds}") as parent_run:
        # 記錄全域資訊（父 run 的 params/tags）
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

            pipe = build_pipeline(args.seed)
            pipe.fit(X_tr, y_tr)

            pred = pipe.predict(X_va)
            acc = accuracy_score(y_va, pred)
            f1 = f1_score(y_va, pred)

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

            # 保存模型與該折的 JSON
            model_path = models_dir / f"fold_{fold}.joblib"
            joblib.dump(pipe, model_path)
            (run_dir / f"fold_{fold}.json").write_text(
                json.dumps(fold_info, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            # 子 run：每折一筆（nested）
            with mlflow.start_run(run_name=f"fold_{fold}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metrics({"acc": acc, "f1": f1})
                mlflow.log_artifact(str(model_path))
                mlflow.log_artifact(str(run_dir / f"fold_{fold}.json"))

        # 平均成績
        avg_acc = float(np.mean([f["metrics"]["acc"] for f in fold_summ]))
        avg_f1  = float(np.mean([f["metrics"]["f1"] for f in fold_summ]))

        # RUN_INFO：寫入檔案並上傳為 artifact（父 run）
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

        mlflow.log_metrics({"avg_acc": avg_acc, "avg_f1": avg_f1})
        mlflow.log_artifact(str(run_info_path))
        # 也把整個 run_dir（所有 fold_*.json）打包上去
        mlflow.log_artifacts(str(run_dir))

    print(f"[OK] 5-fold 完成。平均 acc={avg_acc:.4f}, f1={avg_f1:.4f}")
    print(f"模型輸出：{models_dir.resolve()}")
    print(f"實驗輸出：{run_dir.resolve()}（也同步到 MLflow）")

if __name__ == "__main__":
    main()

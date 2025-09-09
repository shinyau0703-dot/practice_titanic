# src/data/clean.py
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import pandas as pd

KEEP_COLS = [
    # 目標欄位（train 才有）
    "Survived",
    # 常用特徵
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
    # 工程特徵
    "FamilySize",
    # 識別欄位（可保留以便對齊測試集）
    "PassengerId",
]

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

def parse_args():
    p = argparse.ArgumentParser(description="Clean Titanic dataset and export processed CSVs.")
    p.add_argument("--raw-dir", type=str, default="data/raw", help="Directory of raw CSVs")
    p.add_argument("--out-dir", type=str, default="data/processed", help="Output directory for cleaned CSVs")
    return p.parse_args()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _report_na(df: pd.DataFrame, name: str):
    na_cnt = df.isna().sum().sort_values(ascending=False)
    na_pct = (df.isna().mean() * 100).round(2).sort_values(ascending=False)
    logging.info(f"[{name}] missing summary (top 10):\n{pd.DataFrame({'count': na_cnt, 'pct%': na_pct}).head(10)}")

def _family_size(df: pd.DataFrame) -> pd.DataFrame:
    # FamilySize = SibSp + Parch + 1 (自己)
    df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
    return df

def _fill_age_with_median(df: pd.DataFrame, median_age: float | None = None) -> tuple[pd.DataFrame, float]:
    if median_age is None:
        median_age = df["Age"].median()
    df["Age"] = df["Age"].fillna(median_age)
    return df, float(median_age)

def _fill_embarked_with_mode(df: pd.DataFrame, mode_embarked: str | None = None) -> tuple[pd.DataFrame, str]:
    if mode_embarked is None:
        mode_embarked = df["Embarked"].mode(dropna=True)[0]
    df["Embarked"] = df["Embarked"].fillna(mode_embarked)
    return df, str(mode_embarked)

def _drop_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    existing = [c for c in cols if c in df.columns]
    return df.drop(columns=existing)

def clean_split(df: pd.DataFrame, is_train: bool, train_stats: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    清理邏輯：
      - 刪除 Cabin（缺值過多）
      - 建立 FamilySize
      - Age 用中位數補（train 計算；test 使用 train 的統計值）
      - Embarked 用眾數補（同上）
      - 保留常用欄位（KEEP_COLS）
    """
    stats = {} if train_stats is None else dict(train_stats)

    # 1) 刪除 Cabin / 其他高缺失欄位（你也可加上 Ticket、Name 等）
    df = _drop_columns(df, ["Cabin"])

    # 2) 工程特徵
    df = _family_size(df)

    # 3) 補值：Age（中位數）
    if is_train:
        df, median_age = _fill_age_with_median(df, None)
        stats["age_median"] = median_age
    else:
        df, _ = _fill_age_with_median(df, stats.get("age_median"))

    # 4) 補值：Embarked（眾數）
    if is_train:
        df, mode_embarked = _fill_embarked_with_mode(df, None)
        stats["embarked_mode"] = mode_embarked
    else:
        df, _ = _fill_embarked_with_mode(df, stats.get("embarked_mode"))

    # 5) Fare（test 集少數缺值）→ 用中位數補；為一致性，train/test 均以 train 中位數為準
    if "Fare" in df.columns:
        if is_train:
            fare_median = df["Fare"].median()
            stats["fare_median"] = float(fare_median)
            df["Fare"] = df["Fare"].fillna(fare_median)
        else:
            df["Fare"] = df["Fare"].fillna(stats.get("fare_median"))

    # 6) 保留欄位（若是 test 沒有 Survived，自然不會保留）
    keep_cols = [c for c in KEEP_COLS if c in df.columns]
    df = df[keep_cols]

    return df, stats

def main():
    setup_logger()
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    train_path = raw_dir / "train.csv"
    test_path  = raw_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"找不到 raw 檔案：{train_path} 或 {test_path}")

    logging.info(f"讀取：{train_path}")
    train_df = pd.read_csv(train_path)
    logging.info(f"讀取：{test_path}")
    test_df = pd.read_csv(test_path)

    _report_na(train_df, "train(raw)")
    _report_na(test_df, "test(raw)")

    # 清理：先處理 train，統計值用於 test
    train_clean, stats = clean_split(train_df, is_train=True, train_stats=None)
    logging.info(f"Train 清理完成。統計值：{stats}")
    test_clean, _ = clean_split(test_df, is_train=False, train_stats=stats)
    logging.info("Test 清理完成。")

    # 輸出
    out_train = out_dir / "train_clean.csv"
    out_test  = out_dir / "test_clean.csv"
    train_clean.to_csv(out_train, index=False)
    test_clean.to_csv(out_test, index=False)

    logging.info(f"已輸出：{out_train}")
    logging.info(f"已輸出：{out_test}")
    logging.info("Done.")

if __name__ == "__main__":
    main()

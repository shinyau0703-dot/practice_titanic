# src/data/clean.py
"""
Titanic 清理腳本
- 讀取 data/raw/train.csv、data/raw/test.csv
- 進行必要的缺值處理與基本特徵工程
- 輸出到 data/processed/train_clean.csv、data/processed/test_clean.csv
- 回傳/保存訓練集統計值，讓 test 使用相同統計（避免洩漏）
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd


# 只保留這些欄位（test 無 Survived 會自動忽略）
KEEP_COLS = [
    "Survived",                 # 目標（僅 train 有）
    # 常用特徵
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
    # 工程特徵
    "FamilySize",
    # 識別欄位
    "PassengerId",
]


def setup_logger() -> None:
    """設定 logging 格式與等級（在 CLI 與被 app.py 呼叫時保持一致）。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """解析指令列參數：raw 輸入資料夾與 processed 輸出資料夾。"""
    p = argparse.ArgumentParser(description="Clean Titanic dataset and export processed CSVs.")
    p.add_argument("--raw-dir", type=str, default="data/raw", help="Directory of raw CSVs")
    p.add_argument("--out-dir", type=str, default="data/processed", help="Output directory for cleaned CSVs")
    return p.parse_args()


def ensure_dir(p: Path) -> None:
    """若輸出資料夾不存在則遞迴建立。"""
    p.mkdir(parents=True, exist_ok=True)


def _report_na(df: pd.DataFrame, name: str) -> None:
    """將缺值統計（數量與比例，前 10 名）輸出到 log，方便檢視資料品質。"""
    na_cnt = df.isna().sum().sort_values(ascending=False)
    na_pct = (df.isna().mean() * 100).round(2).sort_values(ascending=False)
    logging.info(f"[{name}] missing summary (top 10):\n{pd.DataFrame({'count': na_cnt, 'pct%': na_pct}).head(10)}")


def _family_size(df: pd.DataFrame) -> pd.DataFrame:
    """建立 FamilySize 特徵：SibSp + Parch + 1（自己）。"""
    df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
    return df


def _fill_age_with_median(df: pd.DataFrame, median_age: float | None = None) -> Tuple[pd.DataFrame, float]:
    """用中位數補 Age；若未提供 median_age，則以當前 df 計算並回傳實際使用值。"""
    if median_age is None:
        median_age = df["Age"].median()
    df["Age"] = df["Age"].fillna(median_age)
    return df, float(median_age)


def _fill_embarked_with_mode(df: pd.DataFrame, mode_embarked: str | None = None) -> Tuple[pd.DataFrame, str]:
    """用眾數補 Embarked；若未提供 mode_embarked，則以當前 df 計算並回傳實際使用值。"""
    if mode_embarked is None:
        mode_embarked = df["Embarked"].mode(dropna=True)[0]
    df["Embarked"] = df["Embarked"].fillna(mode_embarked)
    return df, str(mode_embarked)


def _drop_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """安全刪欄：只刪存在的欄位（避免 KeyError）。"""
    existing = [c for c in cols if c in df.columns]
    return df.drop(columns=existing)


def clean_split(
    df: pd.DataFrame,
    is_train: bool,
    train_stats: Dict | None = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    主要清理流程（**此函式不做 I/O**）：
      1) 刪除高缺失欄位（Cabin）；（可按需加入 Ticket/Name 等）
      2) 建立 FamilySize 特徵
      3) Age 用中位數補（train 計算；test 使用 train 的統計）
      4) Embarked 用眾數補（train 計算；test 使用 train 的統計）
      5) Fare 用中位數補（為一致性，test 也用 train 的中位數）
      6) 只保留 KEEP_COLS 中存在的欄位
    回傳：(清理後 DataFrame, 統計值 dict)
    """
    stats = {} if train_stats is None else dict(train_stats)

    # 1) 刪高缺失欄位
    df = _drop_columns(df, ["Cabin"])

    # 2) 工程特徵
    df = _family_size(df)

    # 3) Age 補值（train 計算 → 保存到 stats；test 使用 stats）
    if is_train:
        df, median_age = _fill_age_with_median(df, None)
        stats["age_median"] = median_age
    else:
        df, _ = _fill_age_with_median(df, stats.get("age_median"))

    # 4) Embarked 補值（train 計算 → 保存到 stats；test 使用 stats）
    if is_train:
        df, mode_embarked = _fill_embarked_with_mode(df, None)
        stats["embarked_mode"] = mode_embarked
    else:
        df, _ = _fill_embarked_with_mode(df, stats.get("embarked_mode"))

    # 5) Fare 補值（以 train 中位數為準，避免洩漏）
    if "Fare" in df.columns:
        if is_train:
            fare_median = df["Fare"].median()
            stats["fare_median"] = float(fare_median)
            df["Fare"] = df["Fare"].fillna(fare_median)
        else:
            df["Fare"] = df["Fare"].fillna(stats.get("fare_median"))

    # 6) 保留欄位（test 無 Survived 會自動被忽略）
    keep_cols = [c for c in KEEP_COLS if c in df.columns]
    df = df[keep_cols]

    return df, stats


def run(raw_dir: Path, out_dir: Path) -> Dict:
    """
    封裝的執行函式（**給 app.py 呼叫**）：
    - 讀 raw/train.csv、raw/test.csv
    - 呼叫 clean_split（train → 存 stats；test → 用 stats）
    - 輸出 processed/train_clean.csv、processed/test_clean.csv
    - 回傳 stats（供上層保存或記錄）
    """
    ensure_dir(out_dir)

    train_path = raw_dir / "train.csv"
    test_path = raw_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"找不到 raw 檔案：{train_path} 或 {test_path}")

    logging.info(f"讀取：{train_path}")
    train_df = pd.read_csv(train_path)
    logging.info(f"讀取：{test_path}")
    test_df = pd.read_csv(test_path)

    _report_na(train_df, "train(raw)")
    _report_na(test_df, "test(raw)")

    # 清理：train 先算統計，test 套用相同統計
    train_clean, stats = clean_split(train_df, is_train=True, train_stats=None)
    logging.info(f"Train 清理完成。統計值：{stats}")
    test_clean, _ = clean_split(test_df, is_train=False, train_stats=stats)
    logging.info("Test 清理完成。")

    # 輸出
    out_train = out_dir / "train_clean.csv"
    out_test = out_dir / "test_clean.csv"
    train_clean.to_csv(out_train, index=False)
    test_clean.to_csv(out_test, index=False)

    logging.info(f"已輸出：{out_train}")
    logging.info(f"已輸出：{out_test}")
    return stats


def main() -> None:
    """CLI 進入點：支援直接 `python src/data/clean.py --raw-dir ... --out-dir ...` 執行。"""
    setup_logger()
    args = parse_args()
    stats = run(Path(args.raw_dir), Path(args.out_dir))
    logging.info(f"Done. stats={stats}")


if __name__ == "__main__":
    main()

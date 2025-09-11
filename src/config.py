# src/config.py
from pathlib import Path

# 專案根目錄
ROOT = Path(__file__).resolve().parents[1]

# 資料夾
DATA_DIR        = ROOT / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
LOGS_DIR        = ROOT / "logs"
MODELS_DIR      = ROOT / "models"        # 與 train_kfold.py / app.py 一致
EXPERIMENTS_DIR = ROOT / "experiments"
DOCS_DIR        = ROOT / "docs"

# 檔案路徑
TRAIN_RAW_PATH   = RAW_DIR / "train.csv"
TEST_RAW_PATH    = RAW_DIR / "test.csv"
TRAIN_CLEAN_PATH = PROCESSED_DIR / "train_clean.csv"
TEST_CLEAN_PATH  = PROCESSED_DIR / "test_clean.csv"
MODEL_PATH       = MODELS_DIR / "model.joblib"   # 若需要「單一最新模型」時可用
SUBMISSION_PATH  = ROOT / "submission.csv"

def ensure_directories() -> None:
    """建立專案所需的目錄結構（若不存在）"""
    for p in [RAW_DIR, PROCESSED_DIR, LOGS_DIR, MODELS_DIR, EXPERIMENTS_DIR, DOCS_DIR]:
        p.mkdir(parents=True, exist_ok=True)

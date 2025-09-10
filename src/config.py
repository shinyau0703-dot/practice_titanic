"""
專案設定檔
集中管理所有路徑、常數、隨機種子
"""

from pathlib import Path

# 專案根目錄
PROJECT_ROOT = Path(__file__).parent.parent

# 資料路徑
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_ROOT / "raw"
PROCESSED_DATA_PATH = DATA_ROOT / "processed"

# 原始資料檔案
TRAIN_RAW_PATH = RAW_DATA_PATH / "train.csv"
TEST_RAW_PATH = RAW_DATA_PATH / "test.csv"
SUBMISSION_SAMPLE_PATH = RAW_DATA_PATH / "gender_submission.csv"

# 處理後資料檔案
TRAIN_CLEAN_PATH = PROCESSED_DATA_PATH / "train_clean.csv"
TEST_CLEAN_PATH = PROCESSED_DATA_PATH / "test_clean.csv"

# 模型相關路徑
MODELS_ROOT = PROJECT_ROOT / "models_store"
MODEL_PATH = MODELS_ROOT / "model.joblib"
PREPROCESSOR_PATH = MODELS_ROOT / "preprocessor.joblib"

# 實驗結果路徑
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
BASELINE_RESULTS_PATH = EXPERIMENTS_ROOT / "baseline_results.txt"

# 日誌路徑
LOGS_ROOT = PROJECT_ROOT / "logs"

# 輸出路徑
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"

# 模型參數
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# 模型選擇(LogisticRegression, RandomForest, XGBoost)
DEFAULT_MODEL = "RandomForest" 

# 確保必要資料夾存在
def ensure_directories():
    directories = [
        DATA_ROOT,
        RAW_DATA_PATH,
        PROCESSED_DATA_PATH,
        MODELS_ROOT,
        EXPERIMENTS_ROOT,
        LOGS_ROOT
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("✅ 所有必要資料夾已建立")


# 測試設定
if __name__ == "__main__":
    print("專案設定:")
    print(f"專案根目錄: {PROJECT_ROOT}")
    print(f"訓練資料: {TRAIN_RAW_PATH}")
    print(f"測試資料: {TEST_RAW_PATH}")
    print(f"模型路徑: {MODEL_PATH}")
    print(f"隨機種子: {RANDOM_SEED}")
    ensure_directories()
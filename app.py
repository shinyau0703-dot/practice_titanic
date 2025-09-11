#!/usr/bin/env python3
"""
Titanic 專案主程式
提供 clean/train/eval/predict/pipeline 子命令總控

使用方式:
    python app.py clean        # 清理資料
    python app.py train        # 訓練模型（呼叫 5-fold 程式）
    python app.py eval         # 評估模型
    python app.py predict      # 產生預測結果
    python app.py pipeline     # 完整流程 (clean + train + eval + predict)
"""

import argparse
import sys
import traceback
from pathlib import Path

# 讓 app.py 能 import 到 src/*
sys.path.append(str(Path(__file__).parent))

try:
    from src.config import ensure_directories
    from src.utils import get_logger
except ImportError as e:
    print(f"❌ 無法匯入模組: {e}")
    print("請確認 src/ 資料夾結構是否正確")
    sys.exit(1)


# 設定環境：建立必要資料夾、初始化 logger
def setup_environment():
    """設定環境：建立必要資料夾、設定 logger"""
    ensure_directories()
    logger = get_logger("app")
    logger.info("🚀 Titanic 專案啟動")
    return logger


# 清理資料：讀取 raw/train.csv、raw/test.csv，最小清理後輸出到 processed/
def clean_data():
    """執行資料清理（最小可用版：選欄位、補缺值、輸出至 processed/）"""
    logger = get_logger("clean")
    logger.info("開始執行資料清理...")
    try:
        import pandas as pd
        from src.config import (
            TRAIN_RAW_PATH, TEST_RAW_PATH,
            TRAIN_CLEAN_PATH, TEST_CLEAN_PATH, PROCESSED_DIR
        )

        if not TRAIN_RAW_PATH.exists() or not TEST_RAW_PATH.exists():
            logger.error(f"找不到原始資料：{TRAIN_RAW_PATH} 或 {TEST_RAW_PATH}")
            return False

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        train = pd.read_csv(TRAIN_RAW_PATH)
        test  = pd.read_csv(TEST_RAW_PATH)

        # 這裡示範最小清理（依你前面流程使用的欄位）
        use_cols = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
        train = train[use_cols].copy()
        test  = test[[c for c in use_cols if c != "Survived"]].copy()

        for df in (train, test):
            df["Age"] = df["Age"].fillna(df["Age"].median())
            df["Fare"] = df["Fare"].fillna(df["Fare"].median())
            df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

        train.to_csv(TRAIN_CLEAN_PATH, index=False)
        test.to_csv(TEST_CLEAN_PATH, index=False)

        logger.info(f"✅ 清理完成：{TRAIN_CLEAN_PATH.name}, {TEST_CLEAN_PATH.name}")
        return True
    except Exception as e:
        logger.error(f"❌ 資料清理失敗: {e}", exc_info=True)
        return False


# 模型訓練：以子行程呼叫 5-fold 程式（src.models.train_kfold）
def train_model():
    """執行 5-fold 模型訓練（呼叫 src.models.train_kfold）"""
    logger = get_logger("train")
    logger.info("開始執行模型訓練 (5-fold)...")
    try:
        import subprocess, sys as _sys
        from src.config import TRAIN_CLEAN_PATH

        if not TRAIN_CLEAN_PATH.exists():
            logger.error(f"找不到清理後的訓練資料: {TRAIN_CLEAN_PATH}，請先 clean")
            return False

        cmd = [
            _sys.executable, "-m", "src.models.train_kfold",
            "--data", str(TRAIN_CLEAN_PATH),
            "--target", "Survived",
            "--folds", "5",
            "--seed", "42",
        ]
        logger.info(f"執行：{' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logger.info("✅ 模型訓練完成（模型與實驗已輸出到 models/ 與 experiments/）")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 訓練指令失敗：{e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"❌ 模型訓練失敗：{e}", exc_info=True)
        return False


# 模型評估：載入其中一個 fold 模型，對 train_clean 做基礎評估
def evaluate_model():
    """執行模型評估（示範：載入 fold_0 對訓練資料做簡單 accuracy）"""
    logger = get_logger("evaluate")
    logger.info("開始執行模型評估...")
    try:
        import joblib
        import pandas as pd
        from sklearn.metrics import accuracy_score
        from src.config import TRAIN_CLEAN_PATH

        candidates = sorted(Path("models").glob("fold_*.joblib"))
        if not candidates:
            logger.error("找不到 models/fold_*.joblib，請先 train")
            return False

        model_path = candidates[0]
        clf = joblib.load(model_path)

        df = pd.read_csv(TRAIN_CLEAN_PATH)
        y = df["Survived"].values
        X = df.drop(columns=["Survived"])

        pred = clf.predict(X)
        acc = accuracy_score(y, pred)
        logger.info(f"✅ 基礎評估完成：{model_path.name}，acc={acc:.4f}")
        return True
    except Exception as e:
        logger.error(f"❌ 模型評估失敗: {e}", exc_info=True)
        return False


# 預測：載入其中一個 fold 模型，對 test_clean 輸出 submission.csv
def predict():
    """執行預測並產生提交檔案（示範：載入 fold_0 輸出 submission.csv）"""
    logger = get_logger("predict")
    logger.info("開始執行預測...")
    try:
        import joblib
        import pandas as pd
        from src.config import TEST_CLEAN_PATH, SUBMISSION_PATH

        candidates = sorted(Path("models").glob("fold_*.joblib"))
        if not candidates:
            logger.error("找不到 models/fold_*.joblib，請先 train")
            return False

        model_path = candidates[0]
        clf = joblib.load(model_path)

        test = pd.read_csv(TEST_CLEAN_PATH)
        preds = clf.predict(test)

        # 注意：Titanic 官方需要 PassengerId，若你在清理時保留了此欄位，請改為讀取原始 test 的 PassengerId。
        sub = pd.DataFrame({"PassengerId": range(892, 892 + len(test)), "Survived": preds})
        sub.to_csv(SUBMISSION_PATH, index=False)
        logger.info(f"✅ 預測完成：{SUBMISSION_PATH}")
        return True
    except Exception as e:
        logger.error(f"❌ 預測失敗: {e}", exc_info=True)
        return False


# 一鍵流程：順序執行 clean → train → eval → predict
def run_pipeline():
    """執行完整流程（clean → train → eval → predict）"""
    logger = get_logger("pipeline")
    logger.info("🚀 開始執行完整流程...")

    steps = [
        ("資料清理", clean_data),
        ("模型訓練", train_model),
        ("模型評估", evaluate_model),
        ("產生預測", predict),
    ]

    for step_name, step_func in steps:
        logger.info(f"執行步驟: {step_name}")
        success = step_func()
        if not success:
            logger.error(f"❌ 流程在 '{step_name}' 步驟失敗")
            return False
        logger.info(f"✅ {step_name} 完成")

    logger.info("🎉 完整流程執行完成！")
    return True


# 參數解析與命令派發
def main():
    """主程式入口：解析命令與派發到對應的步驟"""
    parser = argparse.ArgumentParser(
        description="Titanic 專案命令列工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python app.py clean          # 清理原始資料
  python app.py train          # 訓練模型
  python app.py eval           # 評估模型效果
  python app.py predict        # 產生提交檔案
  python app.py pipeline       # 執行完整流程

注意:
  請確保 data/raw/ 資料夾中有 train.csv 和 test.csv
        """,
    )

    parser.add_argument(
        "command",
        choices=["clean", "train", "eval", "predict", "pipeline"],
        help="要執行的命令",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="啟用除錯模式",
    )

    args = parser.parse_args()

    try:
        logger = setup_environment()
        if args.debug:
            logger.info("🐛 除錯模式啟用")

        commands = {
            "clean": clean_data,
            "train": train_model,
            "eval": evaluate_model,
            "predict": predict,
            "pipeline": run_pipeline,
        }

        success = commands[args.command]()
        if success:
            logger.info(f"🎉 命令 '{args.command}' 執行成功")
            sys.exit(0)
        else:
            logger.error(f"❌ 命令 '{args.command}' 執行失敗")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n🛑 使用者中斷執行")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 程式執行錯誤: {e}")
        if args.debug:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

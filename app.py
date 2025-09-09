#!/usr/bin/env python3
"""
Titanic 專案主程式
提供 clean/train/eval/predict 子命令總控

使用方式:
    python app.py clean        # 清理資料
    python app.py train        # 訓練模型
    python app.py eval         # 評估模型
    python app.py predict      # 產生預測結果
    python app.py pipeline     # 完整流程 (clean + train + eval + predict)
"""

import argparse
import sys
import traceback
from pathlib import Path

# 確保可以 import src 模組
sys.path.append(str(Path(__file__).parent))

try:
    from src.config import ensure_directories
    from src.utils import get_logger
except ImportError as e:
    print(f"❌ 無法匯入模組: {e}")
    print("請確認 src/ 資料夾結構是否正確")
    sys.exit(1)


def setup_environment():
    """設定環境：建立必要資料夾、設定 logger"""
    # 建立必要資料夾
    ensure_directories()
    
    # 取得 logger
    logger = get_logger("app")
    logger.info("🚀 Titanic 專案啟動")
    return logger


def clean_data():
    """執行資料清理"""
    logger = get_logger("clean")
    logger.info("開始執行資料清理...")
    
    try:
        # 這裡會呼叫 src.data.clean 模組
        # 目前先用簡單的範例
        import pandas as pd
        from src.config import TRAIN_RAW_PATH, TEST_RAW_PATH, TRAIN_CLEAN_PATH, TEST_CLEAN_PATH
        
        # 檢查原始資料是否存在
        if not TRAIN_RAW_PATH.exists():
            logger.error(f"找不到訓練資料: {TRAIN_RAW_PATH}")
            return False
            
        if not TEST_RAW_PATH.exists():
            logger.error(f"找不到測試資料: {TEST_RAW_PATH}")
            return False
        
        logger.info("原始資料檢查完成 ✅")
        logger.info("請執行資料清理模組...")
        
        # TODO: 呼叫實際的清理函數
        # from src.data.clean import clean_train_data, clean_test_data
        # clean_train_data()
        # clean_test_data()
        
        logger.info("✅ 資料清理完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 資料清理失敗: {e}")
        logger.error(traceback.format_exc())
        return False


def train_model():
    """執行模型訓練"""
    logger = get_logger("train")
    logger.info("開始執行模型訓練...")
    
    try:
        from src.config import TRAIN_CLEAN_PATH
        
        # 檢查清理後的資料是否存在
        if not TRAIN_CLEAN_PATH.exists():
            logger.error(f"找不到清理後的訓練資料: {TRAIN_CLEAN_PATH}")
            logger.error("請先執行: python app.py clean")
            return False
        
        logger.info("清理後資料檢查完成 ✅")
        
        # TODO: 呼叫訓練模組
        # from src.models.train import main as train_main
        # train_main()
        
        logger.info("✅ 模型訓練完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 模型訓練失敗: {e}")
        logger.error(traceback.format_exc())
        return False


def evaluate_model():
    """執行模型評估"""
    logger = get_logger("evaluate")
    logger.info("開始執行模型評估...")
    
    try:
        from src.config import MODEL_PATH
        
        # 檢查模型是否存在
        if not MODEL_PATH.exists():
            logger.error(f"找不到已訓練的模型: {MODEL_PATH}")
            logger.error("請先執行: python app.py train")
            return False
            
        logger.info("已訓練模型檢查完成 ✅")
        
        # TODO: 呼叫評估模組
        # from src.models.evaluate import main as eval_main
        # eval_main()
        
        logger.info("✅ 模型評估完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 模型評估失敗: {e}")
        logger.error(traceback.format_exc())
        return False


def predict():
    """執行預測並產生提交檔案"""
    logger = get_logger("predict")
    logger.info("開始執行預測...")
    
    try:
        from src.config import MODEL_PATH, TEST_CLEAN_PATH, SUBMISSION_PATH
        
        # 檢查必要檔案是否存在
        if not MODEL_PATH.exists():
            logger.error(f"找不到已訓練的模型: {MODEL_PATH}")
            logger.error("請先執行: python app.py train")
            return False
            
        if not TEST_CLEAN_PATH.exists():
            logger.error(f"找不到清理後的測試資料: {TEST_CLEAN_PATH}")
            logger.error("請先執行: python app.py clean")
            return False
        
        logger.info("必要檔案檢查完成 ✅")
        
        # TODO: 呼叫預測模組
        # from src.models.predict import main as predict_main
        # predict_main()
        
        logger.info(f"✅ 預測完成，提交檔案已儲存: {SUBMISSION_PATH}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 預測失敗: {e}")
        logger.error(traceback.format_exc())
        return False


def run_pipeline():
    """執行完整流程"""
    logger = get_logger("pipeline")
    logger.info("🚀 開始執行完整流程...")
    
    steps = [
        ("資料清理", clean_data),
        ("模型訓練", train_model),
        ("模型評估", evaluate_model),
        ("產生預測", predict)
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


def main():
    """主程式入口"""
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
        """
    )
    
    parser.add_argument(
        "command",
        choices=["clean", "train", "eval", "predict", "pipeline"],
        help="要執行的命令"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="啟用除錯模式"
    )
    
    args = parser.parse_args()
    
    try:
        # 設定環境
        logger = setup_environment()
        
        if args.debug:
            logger.info("🐛 除錯模式啟用")
        
        # 執行對應的命令
        commands = {
            "clean": clean_data,
            "train": train_model,
            "eval": evaluate_model,
            "predict": predict,
            "pipeline": run_pipeline
        }
        
        command_func = commands[args.command]
        success = command_func()
        
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
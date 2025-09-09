"""
統一的 logging 設定
提供 get_logger 函數給其他模組使用
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def get_logger(name, log_level=logging.INFO):
    """
    取得設定好的 logger
    
    Args:
        name (str): logger 名稱，通常用 __name__
        log_level: logging 等級，預設 INFO
        
    Returns:
        logging.Logger: 設定好的 logger
    """
    logger = logging.getLogger(name)
    
    # 避免重複加 handler
    if not logger.handlers:
        logger.setLevel(log_level)
        
        # 建立 logs 資料夾
        log_dir = Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # 設定日誌格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 螢幕輸出 (Console Handler)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 檔案輸出 (File Handler)
        log_file = log_dir / "titanic.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 錯誤日誌檔案 (只記錄 WARNING 以上)
        error_log_file = log_dir / "error.log"
        error_handler = logging.FileHandler(error_log_file, encoding='utf-8')
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)
    
    return logger


def log_function_call(func):
    """
    裝飾器：自動記錄函數呼叫
    
    Usage:
        @log_function_call
        def my_function():
            pass
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.info(f"開始執行 {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            logger.info(f"✅ {func.__name__} 執行完成")
            return result
        except Exception as e:
            logger.error(f"❌ {func.__name__} 執行失敗: {e}")
            raise
    
    return wrapper


if __name__ == "__main__":
    # 測試 logger
    logger = get_logger("test")
    
    logger.info("這是一般訊息")
    logger.warning("這是警告訊息")
    logger.error("這是錯誤訊息")
    
    print("✅ Logger 測試完成！")
    print("檢查 logs/ 資料夾是否有 titanic.log 和 error.log")
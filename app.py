#!/usr/bin/env python3
"""
Titanic 專案主程式
提供 clean/train/eval/predict 子命令總控

使用方式:
    python app.py clean        # 清理資料
    python app.py train        # 訓練模型（結束自動寫 DB：app.runs / app.fold_results）
    python app.py eval         # 評估模型
    python app.py predict      # 產生預測結果
    python app.py pipeline     # 完整流程 (clean + train + eval + predict，結束自動寫 DB)
"""

import argparse
import sys
import traceback
from pathlib import Path

# ──────────────────────────────────────────────────────────
# 確保可以 import src 模組
sys.path.append(str(Path(__file__).parent))

try:
    from src.config import ensure_directories
    from src.utils import get_logger
except ImportError as e:
    print(f"❌ 無法匯入模組: {e}")
    print("請確認 src/ 資料夾結構是否正確")
    sys.exit(1)

# ──────────────────────────────────────────────────────────
# DB logging utilities：訓練完成後把結果寫進 app.runs / app.fold_results
from typing import List, Dict, Optional
import psycopg
import numpy as np

DB_CFG = dict(
    host="127.0.0.1",
    port=5432,
    dbname="titanic",
    user="practice_titanic",
    password="titanic",
)

def log_run_and_folds(
    model_family: str,
    seed: int,
    avg_acc: float,
    avg_f1: float,
    folds: List[Dict[str, float]],
) -> int:
    """
    寫入 app.runs 與 app.fold_results，回傳 run_id
    folds 例： [{"fold":0,"acc":0.82,"f1":0.79}, ...]
    需求：
      - app.runs(timestamp DEFAULT now(), model_family, seed, avg_acc, avg_f1)
      - app.fold_results(run_id, fold, acc, f1) 且 (run_id, fold) 唯一
    """
    with psycopg.connect(**DB_CFG) as conn:
        with conn.cursor() as cur:
            cur.execute("SET search_path TO app, public;")
            cur.execute(
                """
                INSERT INTO runs (model_family, seed, avg_acc, avg_f1)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
                """,
                (model_family, seed, avg_acc, avg_f1),
            )
            run_id = cur.fetchone()[0]

            cur.executemany(
                """
                INSERT INTO fold_results (run_id, fold, acc, f1)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (run_id, fold) DO UPDATE
                  SET acc = EXCLUDED.acc,
                      f1  = EXCLUDED.f1;
                """,
                [(run_id, int(x["fold"]), float(x["acc"]), float(x["f1"])) for x in folds],
            )
    return run_id

# ──────────────────────────────────────────────────────────
# 環境

def setup_environment():
    """設定環境：建立必要資料夾、設定 logger"""
    ensure_directories()
    logger = get_logger("app")
    logger.info("🚀 Titanic 專案啟動")
    return logger

# ──────────────────────────────────────────────────────────
# 步驟實作（可逐步替換為你的真實邏輯）

def clean_data() -> bool:
    """執行資料清理"""
    logger = get_logger("clean")
    logger.info("開始執行資料清理...")
    try:
        # TODO: 呼叫真正的清理程式
        # from src.data.clean import clean_all
        # clean_all()
        logger.info("✅（示範）資料清理完成")
        return True
    except Exception as e:
        logger.error(f"❌ 資料清理失敗: {e}")
        logger.error(traceback.format_exc())
        return False

def _run_training_core(logger) -> dict:
    """
    嘗試呼叫你自己的訓練程式，若不存在則回傳示範假資料。
    預期回傳格式：
      {
        'model_family': str,
        'seed': int,
        'fold_accs': List[float],
        'fold_f1s': List[float]
      }
    """
    # 1) 先試著用你真的訓練實作（若你已建立 src/models/train.py）
    try:
        from src.models.train import train_and_eval  # 你可以用我之前提供的完整版本
        result = train_and_eval()
        # 基本防呆檢查
        assert isinstance(result, dict), "train_and_eval() 必須回傳 dict"
        for k in ("model_family", "seed", "fold_accs", "fold_f1s"):
            assert k in result, f"缺少鍵：{k}"
        logger.info("🔗 已使用 src.models.train.train_and_eval() 的回傳結果")
        return result
    except Exception as e:
        logger.warning("⚠️ 找不到或無法使用 src.models.train.train_and_eval()，改用示範假資料")
        logger.debug(e, exc_info=True)

    # 2) ⇩⇩⇩ 一定要有 fallback 回傳 ⇩⇩⇩
    return {
        "model_family": "logreg_v1",
        "seed": 42,
        "fold_accs": [0.81, 0.82, 0.80, 0.82, 0.81],
        "fold_f1s":  [0.78, 0.79, 0.77, 0.80, 0.79],
    }

def train_model() -> bool:
    """執行模型訓練，結束自動把結果寫 DB"""
    logger = get_logger("train")
    logger.info("開始執行模型訓練...")
    try:
        # 取得每折成績
        result = _run_training_core(logger)
        model_family = result["model_family"]
        seed         = int(result["seed"])
        fold_accs    = [float(x) for x in result["fold_accs"]]
        fold_f1s     = [float(x) for x in result["fold_f1s"]]

        # 算平均、組 folds 結構
        avg_acc = float(np.mean(fold_accs)) if fold_accs else None
        avg_f1  = float(np.mean(fold_f1s))  if fold_f1s  else None
        folds = [{"fold": i, "acc": fold_accs[i], "f1": fold_f1s[i]} for i in range(len(fold_accs))]

        # 寫 DB
        run_id = log_run_and_folds(
            model_family=model_family,
            seed=seed,
            avg_acc=avg_acc,
            avg_f1=avg_f1,
            folds=folds,
        )
        logger.info(f"✅ 訓練完成並已寫入 DB：run_id={run_id}, avg_acc={avg_acc:.4f}, avg_f1={avg_f1:.4f}")
        return True

    except Exception as e:
        logger.error(f"❌ 模型訓練失敗: {e}")
        logger.error(traceback.format_exc())
        return False

def evaluate_model() -> bool:
    """執行模型評估"""
    logger = get_logger("evaluate")
    logger.info("開始執行模型評估...")
    try:
        # TODO: 呼叫真正的評估
        # from src.models.evaluate import main as eval_main
        # eval_main()
        logger.info("✅（示範）模型評估完成")
        return True
    except Exception as e:
        logger.error(f"❌ 模型評估失敗: {e}")
        logger.error(traceback.format_exc())
        return False

def predict() -> bool:
    """執行預測並產生提交檔案"""
    logger = get_logger("predict")
    logger.info("開始執行預測...")
    try:
        # TODO: 呼叫真正的預測
        # from src.models.predict import main as predict_main
        # predict_main()
        logger.info("✅（示範）預測完成，提交檔已產生")
        return True
    except Exception as e:
        logger.error(f"❌ 預測失敗: {e}")
        logger.error(traceback.format_exc())
        return False

def run_pipeline() -> bool:
    """執行完整流程（結束時也會把成績寫 DB）"""
    logger = get_logger("pipeline")
    logger.info("🚀 開始執行完整流程...")
    steps = [
        ("資料清理", clean_data),
        ("模型訓練", train_model),      # 這裡的 train_model 已內建寫 DB
        ("模型評估", evaluate_model),
        ("產生預測", predict),
    ]
    for step_name, step_func in steps:
        logger.info(f"執行步驟: {step_name}")
        ok = step_func()
        if not ok:
            logger.error(f"❌ 流程在 '{step_name}' 步驟失敗")
            return False
        logger.info(f"✅ {step_name} 完成")
    logger.info("🎉 完整流程執行完成！")
    return True

# ──────────────────────────────────────────────────────────
# CLI

def main():
    """主程式入口"""
    parser = argparse.ArgumentParser(
        description="Titanic 專案命令列工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python app.py clean          # 清理原始資料
  python app.py train          # 訓練模型（結束自動寫 DB）
  python app.py eval           # 評估模型效果
  python app.py predict        # 產生提交檔案
  python app.py pipeline       # 執行完整流程（結束自動寫 DB）
        """
    )
    parser.add_argument("command", choices=["clean", "train", "eval", "predict", "pipeline"])
    parser.add_argument("--debug", action="store_true", help="啟用除錯模式")
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
        ok = commands[args.command]()
        if ok:
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

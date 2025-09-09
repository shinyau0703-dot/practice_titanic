practice_w1/
├─ app.py                                  (X)  # 入口：clean/train/eval/predict 子命令總控
│
├─ .venv/
├─ .vscode/
│
├─ data/
│  ├─ raw/                                  # 原始 Kaggle Titanic 資料（train.csv, test.csv）
│  └─ processed/                            # 清理後資料（可重生，Git 忽略）
│
├─ docs/
│  ├─ WEEK1_PLAN.md
│  └─ FILES_TREE.md
│
├─ experiments/
│  └─ baseline_results.txt
│
├─ models_store/                            # 存模型/前處理器（joblib / json 等）
│
├─ notebooks/
│  └─ eda.ipynb
│
├─ src/
│  ├─ __init__.py                      (X)  # 讓 src 成為套件（方便 import）
│  ├─ config.py                        (X)  # 集中管理路徑/常數/隨機種子
│  │
│  ├─ utils/
│  │   ├─ __init__.py                 (X)
│  │   └─ logging.py                  (X)  # 統一 logger（get_logger）
│  │
│  ├─ data/
│  │   ├─ __init__.py                 (X)
│  │   └─ clean.py                         # 你已有：資料清理 + 輸出 processed
│  │
│  ├─ features/
│  │   ├─ __init__.py                 (X)
│  │   └─ preprocess.py               (X)  # 建 ColumnTransformer/OneHot/Scaler
│  │
│  └─ models/
│      ├─ __init__.py                 (X)
│      ├─ train.py                         # 你已有：訓練 baseline（建議改成 run_train 介面）
│      ├─ evaluate.py                 (X)  # 載入模型做評估、輸出指標
│      └─ predict.py                  (X)  # 讀 test_clean 產生 submission.csv
│
├─ .gitignore
├─ requirements.txt
└─ README.md

practice_w1/
├─ .venv/               # Python 虛擬環境（本機用，Git 忽略）
├─ .vscode/             # VSCode 專案設定（如 settings.json）
│
├─ data/                # 資料專區
│  ├─ raw/              # 原始 Kaggle Titanic 資料（train.csv, test.csv）
│  ├─ processed/        # 清理後資料（X_train, y_train...，可重生，Git 忽略）
│
├─ docs/                # 文件
│  ├─ WEEK1_PLAN.md     # Week1 計畫（todo list）
│  └─ FILES_NOW_MINI.txt# 檔案清單快照
│
├─ experiments/         # 實驗結果
│  └─ baseline_results.txt   # baseline 模型的報表
│
├─ models_store/        # 訓練好的模型檔（model.joblib, preprocessor.joblib）
│
├─ notebooks/           # Jupyter 筆記本
│  └─ eda.ipynb         # EDA 探索式資料分析
│
├─ src/                 # 專案程式碼
│  ├─ data/             # 資料處理
│  │   └─ clean.py      # 資料清理 + 切分 train/valid
│  ├─ models/           # 模型相關
│  │   └─ train.py      # 訓練 baseline 模型
│  └─ utils/            # 共用工具 (logger, config…)
│
├─ .gitignore           # 忽略清單（.venv/, __pycache__/, data/processed/…）
├─ requirements.txt     # 套件清單（pip freeze 匯出）
└─ README.md            # 專案簡介 + 使用說明 + 檔案樹

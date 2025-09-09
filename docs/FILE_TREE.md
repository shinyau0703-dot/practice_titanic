practice_w1/
├── .venv/                              # Python 虛擬環境（隱藏）
├── .vscode/                            # VS Code 設定（隱藏）
│
│
├── app.py                              # 入口：clean/train/eval/predict 子命令總控
│
├── requirements.txt                    # Python 套件依賴
├── .gitignore                          # Git 忽略設定
├── README.md                           # 專案說明文件
│
│
│
├── data/                           
│   ├── raw/                   # 原始 Kaggle Titanic 資料
│   │   ├── train.csv                   # 訓練資料
│   │   ├── test.csv                    # 測試資料
│   │   └── gender_submission.csv       # 範例提交檔案
│   ├── processed/             # 清理後資料
│   │   ├── train_clean.csv             # 清理後訓練資料
│   │   └── test_clean.csv              # 清理後測試資料
│   └── clean.py               # 資料清理腳本（舊位置）
│
│
├── logs/                      #執行的日誌     
│   ├── titanic.log                     # 主要日誌檔案
│   └── error.log                       # 錯誤日誌檔案
│
│
├── docs/                       #我的筆記     
│   ├── WEEK1_PLAN.md                   # 第一週計畫
│   └── FILES_TREE.md                   # 檔案結構說明
│   └── NOTES                           # 單元小筆記
│
│
├── experiments/               # 模型實驗結果                 
│   └── baseline_results.json           # 基準模型實驗結果
│   └── baseline_results.txt            # 基準模型實驗結果
│
│
├── models_store/                 # 存放訓練好的模型
│   ├── model.joblib                    # 訓練好的模型
│   └── preprocessor.joblib             # [X]前處理器
│
├── notebooks/                    #jupyter notebook
│   └── eda.ipynb                       # 探索性資料分析(EDA)筆記本(可畫圖)
│
└── src/                         # 主要程式碼
    ├── __init__.py                     # 讓 src 成為套件
    ├── config.py                       # 集中管理路徑/常數/隨機種子
    │
    ├── utils/                      
    │   ├── __init__.py             
    │   └── logging.py                  # logger 記錄日誌（get_logger）
    │
    ├── data/                       
    │   ├── __init__.py             
    │   └── clean.py                    # 資料清理模組
    │
    ├── features/                   
    │   ├── __init__.py             
    │   └── preprocess.py           [X] # ColumnTransformer/OneHot/Scaler
    │
    └── models/                     
        ├── __init__.py             
        ├── train.py                    # 訓練 baseline 模型
        ├── evaluate.py                 # 載入模型做評估、輸出指標
        └── predict.py                  # 讀 test_clean 產生 submission.csv
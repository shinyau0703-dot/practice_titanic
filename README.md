# Titanic Week 1 Project

## 專案簡介
這是一個用 Titanic 生存預測 (Kaggle) 來練習 MLOps 基礎的專案。
內容涵蓋：環境管理、資料清理、baseline 模型、重現性測試、Git 版控。

## 檔案清單
```text
titanic_w1/
├─ data/
│  ├─ raw/           # 原始 Kaggle train/test
│  ├─ processed/     # 清理後的資料
├─ notebooks/        # EDA 筆記本
├─ src/
│  ├─ data/          # clean.py 等資料處理程式
│  ├─ models/        # train.py, infer.py
│  └─ utils/         # 工具程式
├─ models_store/     # 訓練好的模型檔
├─ experiments/      # baseline 指標 / 報表
├─ docs/
│  └─ WEEK1_PLAN.md  # 詳細計畫
├─ requirements.txt
├─ .gitignore
└─ README.md

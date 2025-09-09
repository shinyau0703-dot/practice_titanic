# ✅ Week 1 計畫 (Titanic 生存預測)
---
## Day 1 — 把房間建好、工具裝齊、專案架好骨架，上傳到 GitHub

 [V] VScode 連 GitHub

 [V]在 VScode 上建立 venv (虛擬環境)
 未啟用狀態: PS D:\practive_w1>
 成功啟用狀態: (.venv) PS D:\practive_w1>

 [V]安裝需要的套件並存放在 venv
 pip install pandas scikit-learn matplotlib joblib jupyterlab ipykernel python-dotenv

 [V]建立 .gitignore（忽略 venv、pycache、data/processed、models_store）

 [V] 建立資料夾骨架（data, src, models_store, experiments, notebooks）

 [V]下載 Titanic dataset，放到 data/raw/

 [V]撰寫 README/FILE_TREE/WEEK1_PLAN


## Day 2 — 資料前處理(資料清理 + 分割) & baseline 模型訓練

[V] EDA(探索式資料分析)--jupyter notebook

[V] 缺值處理 clean.py

[V] 類別轉數值（OneHot）train.py

[V] 切分 train/valid (80/20)  train.py
     *output-1:models_store/model.joblib
     *output-2:experiments/baseline_results.txt

[V] 使用 Logistic Regression 作 baseline  experiments/baseline_results.txt

[V] 輸出 Accuracy 與分類報表 experiments/baseline_results.txt

[V] 存成 models_store/model.joblib

[V] 記錄結果於 experiments/baseline_results.txt

[ ]flowchart


## Day 3 — 重現性測試

目標：確認專案可從零跑起來

[ ] 模擬新環境：刪掉 venv，重建並安裝 requirements.txt

[ ] 確保能產生相同結果

[ ] 確保能產生相同結果





## Day 4 — 重現性測試



## Day 5 — 專案驗收 & GitHub 更新

目標：完成乾淨可重現專案

 檢查 repo 是否乾淨（不必要檔案都在 .gitignore）

 更新 README（包含：環境建置、資料、清理、訓練、結果檔案位置）

 最後 push GitHub，Week 1 ✅ 完成

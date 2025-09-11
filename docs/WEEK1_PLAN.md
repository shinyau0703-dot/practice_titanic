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


## Day 3 — 重現性

目標：確認專案可從零跑起來&產生相同結果
 
[V] 1環境重現（requirements.txt）

[V] 2資料重現（data/processed 保存清理結果）

[V] 3參數重現（configs.py）

[V] 4隨機性/決定性（random_state）

[V] 5流程重現（python app.py pipeline）

[V] 6模型重現（config）

[V] 7實驗追蹤（kfold,logs紀錄）


## Day 4 — 優化

[ ] 模擬新環境：刪掉 venv，重建並安裝 requirements.txt

[ ] 開發流程 flowchart
*架構為甚麼這樣做(優點?進步?)
*下周目標:下載別人的git 重建虛擬環境 跑他的程式

[V] Git 切換分支
*做完merge回main

[ ] 下周目標:def 不要寫死

目標：給我幾個Git有可能會碰到的問題(練習)
*自動化CI/CD：push 後自動測試程式、訓練模型
*可監控:MLOps pipeline 裡有監控/回訓


## Day 5 — 專案驗收 & GitHub 更新

目標：完成乾淨可重現專案

 檢查 repo 是否乾淨（不必要檔案都在 .gitignore）

 更新 README（包含：環境建置、資料、清理、訓練、結果檔案位置）

 最後 push GitHub，Week 1 ✅ 完成

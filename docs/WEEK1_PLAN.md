# Week 1 計畫 (Titanic 生存預測)

Day 1 — 環境 + GitHub + 專案骨架
目標：建立完整專案結構並 push 到 GitHub
任務：
1. VScode 連 GitHub
2. 在 VScode 上建立 venv (虛擬環境)
   - 目前還沒有啟用虛擬環境: PS D:\practive_w1>
   - 成功啟用虛擬環境: (.venv) PS D:\practive_w1>
   - 將需要的套件都放在這個房間
3. 匯出 requirements.txt
4. 建立 .gitignore（忽略 venv、pycache、data/processed、models_store）
5. 初始化 Git，連 GitHub repo，push 首次 commit
6. 建立資料夾骨架（data, src, models_store, experiments, notebooks）
7. 下載 Titanic dataset，放到 data/raw/
8. 撰寫 README 初版

Day 2 — 資料清理 + 分割
目標：一次完成資料處理流程
任務：
- 決定要用的欄位（Pclass, Sex, Age, SibSp, Parch, Fare, Embarked）
- 缺值處理（Age → 中位數、Embarked → 最常見值）
- 類別轉數值（Sex, Embarked → OneHot）
- 切分 train/valid (80/20)
- 存成 data/processed/ + preprocessor.joblib

Day 3 — baseline 模型訓練
目標：完成第一個可運行的模型
任務：
- 使用 Logistic Regression 作 baseline
- 輸出 Accuracy 與分類報表
- 存成 models_store/model.joblib
- 記錄結果於 experiments/baseline_results.txt

Day 4 — 重現性測試
目標：確認專案可從零跑起來
任務：
- 模擬新環境：刪掉 venv，重建並安裝 requirements.txt
- 依 README 跑 clean.py → train.py
- 確保能產生相同結果
- 修正 README，補充完整流程

Day 5 — 專案驗收 & GitHub 更新
目標：完成乾淨可重現專案
任務：
- 檢查 repo 是否乾淨（不必要檔案都在 .gitignore）
- 更新 README（包含：環境建置、資料、清理、訓練、結果檔案位置）
- 最後 push GitHub，Week 1 ✅ 完成

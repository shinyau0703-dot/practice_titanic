# Week 2 DB Plan（PostgreSQL）
*「安裝 → 建表 → CRUD → 查詢 → 基本索引 → 串 pipeline」

## Day 1：安裝與連線
- [ ] 安裝 PostgreSQL（Docker）
*ctrl + D 跳出  (回到PS D:\practive_w1>)
- [ ] 建立資料庫 titanic
- [ ] 建立使用者 app，設定密碼（只給 titanic 權限）
- [ ] 測試連線：用 psql 或 DBeaver/pgAdmin 連進去，跑 SELECT version();
- [ ] 建立最小 runs 表：id (SERIAL PK), timestamp, model_family, avg_acc, avg_f1
- [ ] 手動插入一筆假 run，確認能查詢出來

## Day 2：匯入與 CRUD
- [ ] 匯入 train_clean.csv 到 PG（COPY 或 Python pandas → to_sql）
- [ ] 練習 CRUD：
  - [ ] C：新增一筆 run（例如 logistic regression，acc=0.78, f1=0.72）
  - [ ] R：查詢最佳 f1 的 run
  - [ ] U：更新一筆 run 的 model_family（測試 update 語法）
  - [ ] D：刪掉一筆 run（熟悉 DELETE 語法）
- [ ] 新增第二張表 fold_results：id, run_id, fold, acc, f1
- [ ] 插入一筆 run_id=1 的 5 筆 fold 資料，並查詢出平均

## Day 3：查詢與索引
- [ ] 練習查詢：
  - [ ] Top 5 平均 f1 的 runs
  - [ ] 同一個 seed 下不同模型的比較
  - [ ] 指定時間區間內（例如今天）的 runs
- [ ] 建立索引：
  - [ ] runs(model_family)
  - [ ] fold_results(run_id)
- [ ] 使用 EXPLAIN，觀察索引有無被使用
- [ ] 匯出查詢結果成 CSV（用 SQL 或 pandas）

## Day 4：整合與自動化
- [ ] 修改 app.py pipeline：
  - [ ] pipeline 執行結束 → 自動寫入 runs 表（avg_acc/f1、seed、model_family）
  - [ ] 每折 fold 的結果 → 寫入 fold_results 表
- [ ] 查詢驗證：pipeline 完成後，能在 PG 裡找到這次 run 的資料
- [ ] 新增小報表腳本（Python）：
  - [ ] 查詢所有 runs → 輸出 all_runs.csv
  - [ ] 查詢歷史最佳 run → 印在 console
- [ ] 嘗試建立一個「失敗 run」：故意插入錯誤資料，觀察 rollback 行為
- [ ] （選擇性）在 logs/ 中輸出 db.log，記錄寫入狀態

- [ ] 整理成pdf

## Bonus（可選挑戰）
- [ ] 使用 pgAdmin/DBeaver GUI 觀察資料表結構
- [ ] 練習備份/還原（pg_dump / pg_restore）
- [ ] 嘗試 JSON 欄位（在 runs 加 feature_flags JSONB，手動插入 {"FamilySize": true}）
- [ ] 嘗試用 Docker 建立第二個 PG container 當「測試環境」

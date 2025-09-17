# Week 2 DB Plan（PostgreSQL）
*「安裝 → 建表 → CRUD → 查詢 → 基本索引 → 串 pipeline」

## Day 1：資料庫作業完成清單 ✅
- [V] 成功透過 SQLTools 連上 PostgreSQL
- [V] 建立了專案資料庫 titanic
- [V] 建立專用使用者 app 並設定密碼
- [V] 建立了最小化的成果表 runs
- [V] 驗證能正常查詢資料

## Day 2：匯入與 CRUD
- [V] 匯入 train_clean.csv 到 PG
- [V] 練習 CRUD(新增/查詢/更新/刪掉)
- [V] 新增第二張表 fold_results：id, run_id, fold, acc, f1
- [V] 插入一筆 run_id=1 的 5 筆 fold 資料，並查詢出平均

## Day 3：查詢與索引
- [V] 練習查詢：
  - [V] Top 5 平均 f1 的 runs
  - [V] 同一個 seed 下不同模型的比較
  - [V] 指定時間區間內（例如今天）的 runs
- [V] 建立索引：
  - [V] runs(model_family)
  - [V] fold_results(run_id)

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













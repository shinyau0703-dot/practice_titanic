Repro Playbook（八大面向）

1) 環境重現（requirements.txt）
做什麼：固定語言版本與依賴，封裝到容器(Docker)。
為什麼：避免「我這台能跑、你那台不能」。

2) 資料重現（data/processed 保存清理結果）
做什麼：原始資料快照、處理後資料版本、結構校驗。
為什麼：資料一變，結果就會變；要能回到「當時那一版」。

3) 參數重現（configs.py）
做什麼：把所有可變參數寫進 YAML config，程式只讀 config。
為什麼：避免「指令列拼參數」遺漏；能用 Git 比對差異。

4) 隨機性/決定性（random_state）
做什麼：固定所有 RNG，必要時採「可重現但可能較慢」的 determinism。
為什麼：隨機抽樣/初始化會影響結果。

5) 流程重現（python app.py pipeline）
做什麼：把「clean → features → train → eval → predict」做成可一鍵跑的工作流。
為什麼：手動步驟容易漏；流水線才能在 CI/CD 或排程器跑。

6) 模型重現（config）
做什麼：保存模型權重 + 前處理器 + 訓練用 config + 指標，並寫入 metadata。
為什麼：同一資料與參數應能得到同一個模型；要能追溯「模型從哪來」。

7) 實驗追蹤（kfold,logs紀錄）
做什麼：每次實驗都產生「誰/何時/用哪個 config/資料版本/commit SHA/結果」的紀錄。
為什麼：比較與回溯。


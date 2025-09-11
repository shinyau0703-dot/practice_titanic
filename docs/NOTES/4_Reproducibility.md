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


<<<<<<< HEAD
5) 流程重現（Pipelines & Orchestration）

做什麼：把「clean → features → train → eval → predict」做成可一鍵跑的工作流。

為什麼：手動步驟容易漏；流水線才能在 CI/CD 或排程器跑。

做法/工具

Makefile / bash / PowerShell：最小可用

進階：Prefect / Airflow / Dagster（排程與監控）

最小示例（Makefile）

.PHONY: all clean preprocess train eval
all: clean preprocess train eval
clean:
	python app.py clean --config configs/baseline.yaml
preprocess:
	python app.py preprocess --config configs/baseline.yaml
train:
	python app.py train --config configs/baseline.yaml
eval:
	python app.py eval --config configs/baseline.yaml

6) 模型重現（Artifacts & Lineage）

做什麼：保存模型權重 + 前處理器 + 訓練用 config + 指標，並寫入 metadata。

為什麼：同一資料與參數應能得到同一個模型；要能追溯「模型從哪來」。

做法/工具

models/ 存 .joblib / .pt 與對應 config.yaml/metrics.json

MLflow/Weights & Biases：自動追蹤參數、指標、工件與 UI 比對

最小示例（存工件與指標）

import json, joblib
joblib.dump(pipeline, "models/model.joblib")
json.dump({"acc": acc, "f1": f1}, open("experiments/2025-09-10_001/metrics.json","w"))

7) 實驗追蹤（MLflow）

做什麼：每次實驗都產生「誰/何時/用哪個 config/資料版本/commit SHA/結果」的紀錄。

為什麼：比較與回溯。


8) 自動化與守門（CI/CD & Tests）

做什麼：push PR 就跑測試與靜態檢查；可選擇自動訓練/評估。

為什麼：避免把壞掉的步驟合進 main；長期可維護。

做法/工具

pytest：對 src/data/clean.py、src/features/preprocess.py、src/models/train.py 寫單元/整合測試

GitHub Actions：裝依賴 → 跑 make all → 上傳 artifacts（模型/報告）

最小示例（.github/workflows/ci.yml）

name: CI
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - run: pytest -q
      - run: make all

你專案的快速升級（可立刻做）

新增 configs/（把超參移出程式）

新增 Makefile（或 run.ps1）一鍵跑 pipeline

在 train 結束時輸出：models/model.joblib + experiments/<run_id>/metrics.json + RUN_INFO.json（含 git commit id、data 版本、config 路徑、seed）

固定種子（如上程式段）

（可選）接 MLflow 做實驗追蹤

（可選）接 DVC 版本化資料

RUN_INFO.json 建議欄位（非常關鍵）
{
  "timestamp": "2025-09-10T12:34:56+08:00",
  "git_commit": "abcdef1",
  "config": "configs/baseline.yaml",
  "seed": 42,
  "python": "3.11.6",
  "packages": "requirements.txt hash: sha256:...",
  "data": {
    "raw": "dvc md5: 3c45... or file hash",
    "processed": "sha256: 9ab1..."
  },
  "hardware": { "cuda": false, "cpu": "i7-12700H" }
}

迷你參考流程（CLI）
# 準備環境
python -m venv .venv && .\.venv\Scripts\activate
python -m pip install -U pip -r requirements.txt

# 一鍵跑
make all
# 或
python app.py clean   --config configs/baseline.yaml
python app.py train   --config configs/baseline.yaml
python app.py eval    --config configs/baseline.yaml
python app.py predict --config configs/baseline.yaml

一句話總結

重現性 = 環境 + 資料 + 參數 + 隨機性 + 流程 + 模型工件 + 實驗追蹤 + 自動化。
你現在的目錄結構已經打好地基；照上面 Playbook 逐步加上 Config 化、Seed 固定、Artifacts 與 Run 記錄、Makefile、（可選）MLflow/DVC，你的專案就具備工程級可重現的能力。

如果你願意，我可以直接幫你把 configs/、Makefile、RUN_INFO 紀錄程式碼 做成 PR 級別的檔案片段，貼進你的 repo 就能用。
=======
>>>>>>> 4e37e08 (main)

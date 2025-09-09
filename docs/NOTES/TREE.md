# Titanic 專案流程圖

```mermaid
flowchart TD
  %% === 主幹線：app.py 控制流程靠左 ===
  L[app.py<br/>clean / train / eval / predict]:::entry
  L -->|呼叫清理| B[資料清理<br/>src/data/clean.py]:::code
  B -->|產生處理後資料| R2[data/processed<br/>train_clean.csv / test_clean.csv]:::data
  R2 -->|送進前處理| C[特徵前處理<br/>features/preprocess.py]:::code
  C -->|提供特徵| E[模型訓練<br/>models/train.py]:::code
  E -->|輸出模型| F[models_store/model.joblib]:::artifact
  F -->|載入模型| H[模型評估<br/>models/evaluate.py]:::code
  F -->|載入模型| J[模型預測<br/>models/predict.py]:::code
  J -->|產生提交檔| K[submission.csv]:::artifact

  %% === 支線 ===
  C -->|儲存前處理器| D[models_store/preprocessor.joblib]:::artifact
  D -->|套用規則| E
  E -->|紀錄實驗結果| G[experiments/baseline_results.*]:::artifact
  D -->|載入前處理器| H
  R2 -->|測試資料| H
  H -->|輸出日誌| I[logs/titanic.log / error.log]:::log
  D -->|載入前處理器| J
  R2 -->|測試資料| J

  %% === 文件與設定 ===
  M[src/config.py]:::config -->|提供設定| E & H & J
  O[notebooks/eda.ipynb]:::doc -->|EDA 協助清理| B
  N[docs/*]:::doc

  %% === 樣式定義 ===
  classDef entry fill:#1f77b4,stroke:#0d3d63,color:#fff
  classDef code fill:#e8f0fe,stroke:#6b8bd6,color:#1b3a7a
  classDef data fill:#eaf7ea,stroke:#66a366,color:#1f4d1f
  classDef artifact fill:#fff4e6,stroke:#d49a3a,color:#6b4b16
  classDef log fill:#fde8ef,stroke:#d66b8b,color:#7a1b3a
  classDef config fill:#f0efff,stroke:#8a86d6,color:#2d2a7a
  classDef doc fill:#f7f7f7,stroke:#bdbdbd,color:#4d4d4d

  %% === 主幹線箭頭樣式（紅色粗體） ===
  linkStyle 0 stroke:red,stroke-width:3px
  linkStyle 1 stroke:red,stroke-width:3px
  linkStyle 2 stroke:red,stroke-width:3px
  linkStyle 3 stroke:red,stroke-width:3px
  linkStyle 4 stroke:red,stroke-width:3px
  linkStyle 5 stroke:red,stroke-width:3px



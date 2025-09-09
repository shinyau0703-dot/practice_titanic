# 🔍 Titanic 專案筆記 — 前處理常用套件 & EDA 流程
**EDA = Exploratory Data Analysis，探索式資料分析
---

### [前處理- clean.py]架構

# 🧹 Titanic `clean.py` 架構與資料流

## 1. 函式架構圖

clean.py
├─ 常數
│ └─ KEEP_COLS # 清理後要保留的欄位清單
│
├─ 基礎工具
│ ├─ setup_logger() # 設定 logging 格式
│ ├─ parse_args() # 讀取命令列參數 (--raw-dir, --out-dir)
│ └─ ensure_dir() # 建立資料夾（若不存在）
│
├─ 輔助函式（純資料轉換，不做 I/O）
│ ├─ _report_na() # 缺值統計，寫 log
│ ├─ _family_size() # 建立 FamilySize 特徵
│ ├─ _fill_age_with_median() # 用中位數補 Age
│ ├─ _fill_embarked_with_mode()# 用眾數補 Embarked
│ └─ _drop_columns() # 刪除指定欄位（存在才刪）
│
├─ clean_split() # 核心清理流程（不做 I/O，回傳 df, stats）
│ ├─ 呼叫 _drop_columns()
│ ├─ 呼叫 _family_size()
│ ├─ 呼叫 _fill_age_with_median()
│ ├─ 呼叫 _fill_embarked_with_mode()
│ ├─ 填補 Fare
│ └─ 篩選 KEEP_COLS
│
├─ run() # 封裝流程（含 I/O）
│ ├─ ensure_dir()
│ ├─ 讀 raw/train.csv, raw/test.csv
│ ├─ _report_na()
│ ├─ clean_split(train, is_train=True)
│ ├─ clean_split(test , is_train=False, 使用 train 統計值)
│ ├─ 輸出 train_clean.csv, test_clean.csv
│ └─ return stats
│
└─ main() # CLI 入口
├─ setup_logger()
├─ parse_args()
├─ run(raw_dir, out_dir)
└─ log 完成訊息


---

## 2. I/O 資料流圖

命令列 / app.py
│
▼
main()
│
▼
run()
│
├─ ensure_dir()
├─ 讀取 ● raw/train.csv
├─ 讀取 ● raw/test.csv
├─ _report_na(train/test)
│
├─ clean_split(train, is_train=True)
│ → 輸出 train_clean_df
│ → 產生 ◆ stats (age_median, embarked_mode, fare_median)
│
├─ clean_split(test, is_train=False, 使用 ◆ stats)
│ → 輸出 test_clean_df
│
├─ 寫檔 ● processed/train_clean.csv
├─ 寫檔 ● processed/test_clean.csv
│
└─ return ◆ stats → main() → (給 app.py 或 log 使用)

---

## 3. 圖例說明

- **●** = 檔案 (I/O)  
- **◆** = 統計值（由 train 計算，給 test 使用）  
- **clean_split()** = 純資料轉換，不直接做 I/O  

---

## 4. 📦 前處理常用套件整理

### 🟦 資料操作 / 基礎工具
- **pandas**  
  表格資料處理（讀寫 CSV、DataFrame 運算、缺值處理、合併等）
- **numpy**  
  底層數值計算（陣列、矩陣、統計函數）
- **os / pathlib**  
  路徑與檔案處理，確認目錄與檔案存在性

### 🟩 缺值處理 / 特徵工程
- **scikit-learn (`sklearn`)**
  - `sklearn.impute` :：均值 / 中位數 / 眾數填補   
  - `sklearn.preprocessing`  

### 🟨 資料視覺化
- **matplotlib**  
  基礎繪圖：直方圖、散點圖、條狀圖  

### 🟥 實驗追蹤（未用到）
- **mlflow**  
  實驗管理，記錄參數、指標與模型  
- **wandb (Weights & Biases)**  
  雲端實驗追蹤，適合團隊合作  


---
---
---




### [EDA- eda_titanic.ipynb]架構

## 📊 Titanic  流程

### Step 0 — 基本設定
- 確認工作目錄 (CWD) 是否正確  
- 確認 Python 版本是否來自虛擬環境 `.venv`  
- 檔案路徑是否能正確找到 `train.csv`、`test.csv`

### Step 1 — 匯入與讀檔
### Step 2 — 初步檢查
- 查看資料列數與欄位數  
- 使用 `head()` 檢視前幾筆資料  
- 確認欄位清單  

### Step 3 — 資料結構 & 缺值檢查
- 使用 `.info()` 確認欄位型態（數值/類別）  
- 使用 `.isna().sum()` 計算缺值數量  
- 計算缺值比例，判斷哪些欄位需要處理或捨棄  

### Step 4 — 單變數分佈 (Univariate Analysis)
- 數值型變數：`Age`、`Fare` → 畫直方圖  
- 類別型變數：`Sex`、`Pclass`、`Embarked` → 計算人數並繪製分布圖  

### Step 5 — 雙變數關聯 (Bivariate Analysis)
- `Sex` vs `Survived` → 生存率與性別  
- `Pclass` vs `Survived` → 生存率與艙等  
- `Age` 分箱 vs `Survived` → 生存率與年齡群組  
- `Fare` 分組 vs `Survived` → 生存率與票價分佈  

### Step 6 — 多變數交互作用
- `Sex + Pclass` → 性別與艙等的聯合影響  
- `SibSp + Parch → FamilySize` → 家庭人數與生存率關係  

### Step 7 — 總結
- 保留重要特徵：`Sex`, `Pclass`, `Age`, `Fare`, `FamilySize`  
- 考慮刪除或簡化：`Cabin`, `Ticket`  
- 類別型需轉換：`Sex`, `Embarked`  
- 數值型可分箱或標準化：`Age`, `Fare`  

---









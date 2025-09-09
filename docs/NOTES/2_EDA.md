# 🔍 Titanic 專案筆記 — 前處理常用套件 & EDA 流程
**EDA = Exploratory Data Analysis，探索式資料分析
---

## 📦 前處理常用套件整理

### 🟦 資料操作 / 基礎工具
- **pandas**  
  表格資料處理（讀寫 CSV、DataFrame 運算、缺值處理、合併等）
- **numpy**  
  底層數值計算（陣列、矩陣、統計函數）
- **os / pathlib**  
  路徑與檔案處理，確認目錄與檔案存在性

### 🟩 缺值處理 / 特徵工程
- **scikit-learn (`sklearn`)**
  - `sklearn.impute`  
    - `SimpleImputer`：均值 / 中位數 / 眾數填補  
    - `KNNImputer`：用鄰近樣本補值  
  - `sklearn.preprocessing`  
    - `StandardScaler`：標準化 (平均=0, 方差=1)  
    - `MinMaxScaler`：縮放到 [0,1]  
    - `OneHotEncoder`：類別變數轉數值  
    - `LabelEncoder`：將類別轉為整數編碼  
    - `PolynomialFeatures`：生成多項式特徵  
  - `sklearn.feature_selection`  
    - `SelectKBest`、`RFE`：特徵選擇工具

### 🟨 資料視覺化
- **matplotlib**  
  基礎繪圖：直方圖、散點圖、條狀圖  
- **seaborn**  
  統計視覺化：heatmap、boxplot、countplot、barplot  

### 🟧 進階處理
- **scipy**  
  統計方法（t-test、卡方檢定、常態檢驗等）  
- **category_encoders**  
  進階編碼（Target Encoding、Ordinal Encoding 等）  
- **imbalanced-learn (`imblearn`)**  
  資料不平衡處理（SMOTE、RandomUnderSampler 等）

### 🟥 實驗追蹤（可選）
- **mlflow**  
  實驗管理，記錄參數、指標與模型  
- **wandb (Weights & Biases)**  
  雲端實驗追蹤，適合團隊合作  

---

## 📊 Titanic EDA 流程

### Step 0 — 基本設定
- 確認工作目錄 (CWD) 是否正確  
- 確認 Python 版本是否來自虛擬環境 `.venv`  
- 檔案路徑是否能正確找到 `train.csv`、`test.csv`

### Step 1 — 匯入與讀檔
- 讀取 `train.csv`（訓練資料）  
- 讀取 `test.csv`（測試資料）  
- 讀取 `gender_submission.csv`（Kaggle 提交範例，可選）  
- 確認資料大小與欄位名稱  

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

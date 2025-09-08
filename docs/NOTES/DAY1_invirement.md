# 🚀 Day1 筆記本 — 環境建立與專案骨架

---

## 🏠 虛擬環境 (venv)
- **觀念**：就像專案的「獨立小房間」，避免不同專案的套件互相打架。
- **建立**
python -m venv .venv

markdown
複製程式碼
- **啟用**
& ..venv\Scripts\Activate.ps1

markdown
複製程式碼
✔ 提示字首會變成：`(.venv) PS ...`
- **確認是不是用對 Python**
python -c "import sys; print(sys.executable)"

r
複製程式碼
✔ 應該看到：`D:\practive_w1\.venv\Scripts\python.exe`
- **關閉房間**
deactivate

yaml
複製程式碼

---

## 🛠 安裝套件
- **為什麼要裝？**  
就是專案的工具箱。
- **常用工具箱**
pip install pandas scikit-learn matplotlib joblib jupyterlab ipykernel python-dotenv

markdown
複製程式碼
- **匯出清單（留給未來或別人重建環境用）**
pip freeze > requirements.txt

yaml
複製程式碼

---

## 📚 Git & GitHub
- **用途**：幫你記錄每一步，可以回到舊版本，也能分享。
- **初始化（只要做一次）**
git init
git remote add origin https://github.com/<帳號>/<repo>.git

markdown
複製程式碼
- **日常三連發**
git status # 看目前狀況
git add . # 把改動放到待提交
git commit -m "訊息" # 建立一個版本
git push origin main # 上傳到 GitHub

markdown
複製程式碼
- **拉最新版本，避免衝突**
git pull --rebase origin main

markdown
複製程式碼
- **.gitignore：不要上傳的東西**
.venv/
pycache/
data/processed/
models_store/

yaml
複製程式碼

---

## 📂 專案骨架
乾淨的結構 → 未來維護容易。

practice_w1/
├─ .venv/ # 虛擬環境（房間，不上傳）
├─ data/
│ ├─ raw/ # 原始資料 (Titanic train/test.csv)
│ ├─ processed/ # 清理後資料（不上傳）
├─ docs/
│ ├─ WEEK1_PLAN.md # 計畫
│ ├─ FILE_TREE.md # 檔案結構快照
│ └─ NOTES/ # 學習筆記
├─ experiments/ # 實驗結果（報表/圖）
├─ models_store/ # 訓練好的模型（不上傳）
├─ notebooks/ # Jupyter EDA 筆記本
├─ src/ # 程式碼 (clean.py, train.py …)
├─ requirements.txt # 套件清單
├─ .gitignore
└─ README.md

yaml
複製程式碼

---

## 📝 常見小狀況
- **推送失敗**
git pull --rebase origin main
git push

markdown
複製程式碼
- **push 很慢** → 檢查是不是大檔案沒忽略  
（解法：.gitignore + `git rm --cached`）
- **終端機出現 `>>` 卡住** → 代表指令還沒結束，用 **Ctrl + C** 跳出
- **想看 Git 目前追蹤哪些檔案**
git ls-files

yaml
複製程式碼

---

✅ **今天完成**
- 建好虛擬環境 & 套件  
- 專案骨架整理好  
- GitHub 成功連線  
- README / FILE_TREE / WEEK1_PLAN 都準備好
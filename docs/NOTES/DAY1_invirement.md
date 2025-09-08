# 🚀 Day1 總結筆記 — 環境建立與專案骨架

## 1. 虛擬環境 venv
就像「專案的獨立小房間」，避免不同專案套件打架。

**建立：**
```powershell
python -m venv .venv
啟用：

powershell
複製程式碼
& .\.venv\Scripts\Activate.ps1
→ 成功會看到提示字首變成 (.venv) PS ...

確認使用的 Python 是這個房間：

powershell
複製程式碼
python -c "import sys; print(sys.executable)"
→ 應該是 D:\practive_w1\.venv\Scripts\python.exe

關閉：

powershell
複製程式碼
deactivate
2. 安裝套件
工具箱安裝到這個 venv 裡。

安裝常用套件：

powershell
複製程式碼
pip install pandas scikit-learn matplotlib joblib jupyterlab ipykernel python-dotenv
匯出清單（方便別人重建環境）：

powershell
複製程式碼
pip freeze > requirements.txt
3. Git & GitHub
版本控制：記錄專案每一步，能回溯、能分享。

初始化（只做一次）：

powershell
複製程式碼
git init
git remote add origin https://github.com/<你帳號>/<repo>.git
常用流程：

powershell
複製程式碼
git status               # 看目前狀態
git add <檔案或資料夾>   # 加入待提交
git commit -m "訊息"     # 建立提交
git push origin main     # 上傳到 GitHub
git pull --rebase origin main   # 拉最新，避免衝突
檔案不要上傳（例如大資料、venv） → 寫在 .gitignore：

bash
複製程式碼
.venv/
__pycache__/
data/processed/
models_store/
4. 專案骨架
清楚的資料夾結構，未來維護容易。

bash
複製程式碼
practice_w1/
├─ .venv/             # 虛擬環境（本地用，不上傳）
├─ data/
│  ├─ raw/            # 原始資料 (Titanic train/test.csv)
│  ├─ processed/      # 清理後資料（Git 忽略）
├─ docs/
│  ├─ WEEK1_PLAN.md   # 計畫
│  ├─ FILE_TREE.md    # 專案結構快照
├─ experiments/       # 實驗結果（報表、圖）
├─ models_store/      # 模型檔案（Git 忽略）
├─ notebooks/         # Jupyter EDA 筆記本
├─ src/               # 程式碼 (clean.py, train.py …)
├─ requirements.txt   # 套件清單
├─ .gitignore
└─ README.md
5. 常見狀況小抄
推送失敗 →

powershell
複製程式碼
git pull --rebase origin main
git push
push 很慢 → 檢查是不是不小心把大檔追蹤了
（用 .gitignore + git rm --cached 移除）

終端機卡住 >> → 代表指令沒打完，用 Ctrl+C 中止

想知道 Git 追蹤了什麼：

powershell
複製程式碼
git ls-files
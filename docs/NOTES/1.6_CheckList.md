## Check List

# A) Repo 乾淨度

[V] 工作樹乾淨（無遺漏變更）
驗證：
git status --porcelain
輸出應為空（沒有任何列）。

[V] 未追蹤檔僅為預期（如問題筆記、臨時檔）
驗證：
git ls-files -o --exclude-standard
確認列表沒有應該被追蹤的檔案（如 src/*.py、README.md）。

[V] .gitignore 覆蓋常見髒檔（.venv、pycache、data/raw…）
驗證（抽樣幾個路徑）：
git check-ignore -v .venv/ src/__pycache__/ data/raw/ models_store/ outputs/ reports/figures/
每個路徑都應回報被哪一條規則忽略。若無輸出＝沒忽略，需補 .gitignore。

[V] 沒有大檔被追蹤（>50MB 建議用 LFS 或忽略）
驗證：
git ls-files | ForEach-Object {
  $f=Get-Item $_
  if($f.Length -gt 50MB){ '{0}  {1:N1} MB' -f $f.FullName, ($f.Length/1MB) }
}
無輸出＝OK；有輸出→改用 .gitignore 或 Git LFS。

# B) 可重現環境

[V] requirements.txt（或 environment.yml）存在且版本已釘死
驗證（pip 方案）：
Test-Path requirements.txt
Select-String -Path requirements.txt -Pattern '==' | Measure-Object | % Count
需存在且多數套件為 == 釘版本。

[V] 能從零重建虛擬環境並成功安裝
驗證（模擬重建）：
*若不想刪現有 .venv，請改用另一個資料夾名
Remove-Item -Recurse -Force .venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
安裝應無 error。

[V] 統一隨機種子（若有用到 ML/統計）
驗證（檢查工具檔）：
確認 src/utils/seed.py 存在，且在主要進入點有呼叫。
Test-Path src/utils/seed.py
Select-String -Path src/**/*.py -Pattern 'from src\.utils\.seed import|import .*seed'




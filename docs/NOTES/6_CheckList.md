A) Repo 乾淨度

 工作樹乾淨（無遺漏變更）
驗證：

git status --porcelain


輸出應為空（沒有任何列）。

 未追蹤檔僅為預期（如問題筆記、臨時檔）
驗證：

git ls-files -o --exclude-standard


確認列表沒有應該被追蹤的檔案（如 src/*.py、README.md）。

 .gitignore 覆蓋常見髒檔（.venv、pycache、data/raw…）
驗證（抽樣幾個路徑）：

git check-ignore -v .venv/ src/__pycache__/ data/raw/ models_store/ outputs/ reports/figures/


每個路徑都應回報被哪一條規則忽略。若無輸出＝沒忽略，需補 .gitignore。

 沒有大檔被追蹤（>50MB 建議用 LFS 或忽略）
驗證：

git ls-files | ForEach-Object {
  $f=Get-Item $_
  if($f.Length -gt 50MB){ '{0}  {1:N1} MB' -f $f.FullName, ($f.Length/1MB) }
}


無輸出＝OK；有輸出→改用 .gitignore 或 Git LFS。

B) 可重現環境

 requirements.txt（或 environment.yml）存在且版本已釘死
驗證（pip 方案）：

Test-Path requirements.txt
Select-String -Path requirements.txt -Pattern '==' | Measure-Object | % Count


需存在且多數套件為 == 釘版本。

 能從零重建虛擬環境並成功安裝
驗證（模擬重建）：

# ⚠️ 若不想刪現有 .venv，請改用另一個資料夾名
Remove-Item -Recurse -Force .venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt


安裝應無 error。

 統一隨機種子（若有用到 ML/統計）
驗證（檢查工具檔）：
確認 src/utils/seed.py 存在，且在主要進入點有呼叫。

Test-Path src/utils/seed.py
Select-String -Path src/**/*.py -Pattern 'from src\.utils\.seed import|import .*seed'

C) 資料與流程

 資料分區結構到位（raw/interim/processed/sample）
驗證：

('data/raw','data/interim','data/processed','data/sample') | % { Test-Path $_ | %{ "$_ => $($_)" } }


raw/interim/processed 應存在且被 .gitignore 忽略（用 A-3 驗證）。

 清理腳本可執行且產出
驗證：

python -m src.data.clean --help
python -m src.data.clean --input data/raw --output data/interim
Test-Path data/interim/cleaned.parquet


 訓練腳本可執行且產出到忽略目錄
驗證：

python -m src.models.train --help
python -m src.models.train --data data/interim/cleaned.parquet --out models_store/run_test/
Test-Path models_store/run_test
git check-ignore -v models_store/

D) 文件（README 與說明）

 README 含 5 大段落（環境建置、資料、清理、訓練、結果）
驗證（關鍵字檢查）：

Select-String -Path README.md -Pattern '#','環境','資料','清理','訓練','結果'


這些字樣都應出現（或英文對應：Environment/Data/Clean/Train/Results）。

 data/README.md 說明資料來源與欄位
驗證：

Test-Path data/README.md


 已知議題（如 Hub 貼 JPG 失敗）已記錄
驗證（Issues 或文件）：

Select-String -Path README.md,docs/**/*.md -Pattern 'Hub','JPG','issue','已知議題'

E) 程式品質與規範（選擇性強烈建議）

 linters/formatter 通過（black/ruff/isort 或等效）
驗證（若用 pre-commit）：

pre-commit run --all-files


無失敗即 OK。

 Conventional Commits
驗證（最近 10 筆）：

git log -10 --pretty=format:"%s"


應看到 feat:, fix:, docs:, chore: 等前綴。

F) 最終提交與遠端

 所有變更已加入並提交
驗證：

git add .
git commit -m "chore(repo): week1 cleanup, reproducible structure, docs"
git status --porcelain


應為空。

 已推送到 origin/main
驗證：

git push origin main
git status


應顯示「Your branch is up to date with 'origin/main'.」

 （可選）建立 Week1 標籤
驗證：

git tag -a week1 -m "Week 1 complete: clean & reproducible"
git push origin week1
git tag --list | Select-String week1

小抄：安全清理指令差異

只刪未追蹤且未被忽略的檔案（較安全）

git clean -fdn   # 預覽
git clean -fd    # 執行


連被忽略的也一起刪（最乾淨，謹慎）

git clean -xfdn  # 預覽
git clean -xfd   # 執行
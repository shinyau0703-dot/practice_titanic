1. 首先檢查必要檔案是否存在：
powershell# 檢查主要檔案
dir app.py, requirements.txt, README.md

# 檢查 src 結構
tree src /F
2. 測試各模組是否能正常 import：
powershell# 測試 config 模組
python -c "from src.config import PROJECT_ROOT; print(f'專案根目錄: {PROJECT_ROOT}')"

# 測試 utils 模組
python -c "from src.utils import get_logger; logger = get_logger('test'); print('Logger 正常')"

# 測試 features 模組
python -c "from src.features import get_preprocessor; print('Features 正常')"
3. 檢查必要的 Python 套件：
powershell# 檢查是否有 requirements.txt
type requirements.txt

# 檢查已安裝的套件
pip list | Select-String "pandas|sklearn|numpy"
4. 測試 app.py 主程式：
powershell# 測試幫助功能
python app.py --help

# 測試設定檔功能
python -m src.config
5. 檢查資料夾結構完整性：
powershell# 檢查重要資料夾是否存在
@("data\raw", "data\processed", "models_store", "logs", "experiments") | ForEach-Object {
    if (Test-Path $_) { 
        Write-Host "✅ $_ 存在" 
    } else { 
        Write-Host "❌ $_ 不存在" 
    }
}
6. 如果要讓別人重現，需要：
建立 requirements.txt：
powershellpip freeze > requirements.txt
建立 README.md：
powershellNew-Item -ItemType File -Path "README.md"
"""
特徵前處理模組
負責建立 ColumnTransformer，處理類別變數編碼、數值變數縮放等
支援訓練和預測階段的資料前處理
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path


def get_preprocessor(use_advanced_features=True):
    """
    建立並回傳前處理管線 (ColumnTransformer)
    
    Args:
        use_advanced_features (bool): 是否使用進階特徵（FamilySize, Title）
        
    Returns:
        ColumnTransformer: 完整的前處理管線
    """
    
    if use_advanced_features:
        # 包含衍生特徵的版本
        numeric_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
        categorical_features = ['Sex', 'Embarked', 'Pclass', 'Title']
    else:
        # 基本特徵版本（適用於剛開始的資料）
        numeric_features = ['Age', 'Fare', 'SibSp', 'Parch']
        categorical_features = ['Sex', 'Embarked', 'Pclass']
    
    # 數值型欄位的處理管線
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # 用中位數填補遺失值
        ('scaler', StandardScaler())                    # 標準化
    ])
    
    # 類別型欄位的處理管線  
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # 用最常見值填補
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    # 組合成 ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # 其他欄位都丟掉
    )
    
    return preprocessor


def create_basic_features(df):
    """
    建立基本的衍生特徵
    
    Args:
        df (pd.DataFrame): 輸入資料
        
    Returns:
        pd.DataFrame: 加上衍生特徵的資料
    """
    df = df.copy()
    
    # 1. FamilySize: 家庭人數
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # 2. IsAlone: 是否獨自一人
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # 3. Title: 從姓名提取稱謂
    if 'Name' in df.columns:
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        
        # 簡化稱謂分類
        title_mapping = {
            'Mr': 'Mr',
            'Miss': 'Miss', 
            'Mrs': 'Mrs',
            'Master': 'Master',
            'Dr': 'Rare',
            'Rev': 'Rare',
            'Col': 'Rare',
            'Major': 'Rare',
            'Mlle': 'Miss',
            'Countess': 'Rare',
            'Ms': 'Miss',
            'Lady': 'Rare',
            'Jonkheer': 'Rare',
            'Don': 'Rare',
            'Dona': 'Rare',
            'Mme': 'Mrs',
            'Capt': 'Rare',
            'Sir': 'Rare'
        }
        
        df['Title'] = df['Title'].map(title_mapping).fillna('Rare')
    
    # 4. Age groups: 年齡分組
    if 'Age' in df.columns:
        df['AgeGroup'] = pd.cut(df['Age'], 
                               bins=[0, 12, 18, 35, 60, np.inf], 
                               labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
        df['AgeGroup'] = df['AgeGroup'].astype(str)
    
    # 5. Fare groups: 票價分組
    if 'Fare' in df.columns:
        df['FareGroup'] = pd.qcut(df['Fare'].fillna(df['Fare'].median()), 
                                 q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'])
        df['FareGroup'] = df['FareGroup'].astype(str)
    
    return df


def preprocess_data(df, preprocessor=None, is_training=True, use_advanced_features=True):
    """
    使用 ColumnTransformer 前處理資料
    
    Args:
        df (pd.DataFrame): 輸入資料
        preprocessor: 已訓練的前處理器，如果是 None 會建立新的
        is_training (bool): 是否為訓練資料（決定要不要 fit）
        use_advanced_features (bool): 是否使用進階特徵
        
    Returns:
        tuple: (處理後的特徵矩陣, 前處理器, 特徵名稱)
    """
    
    # 建立衍生特徵
    if use_advanced_features:
        df_processed = create_basic_features(df)
    else:
        df_processed = df.copy()
    
    # 建立或使用現有的前處理器
    if preprocessor is None:
        preprocessor = get_preprocessor(use_advanced_features)
    
    if is_training:
        # 訓練時：fit_transform
        X_processed = preprocessor.fit_transform(df_processed)
        print(f"✅ 訓練資料前處理完成，形狀: {X_processed.shape}")
    else:
        # 測試時：只 transform
        X_processed = preprocessor.transform(df_processed)
        print(f"✅ 測試資料前處理完成，形狀: {X_processed.shape}")
    
    # 取得特徵名稱
    feature_names = get_feature_names(preprocessor)
    
    return X_processed, preprocessor, feature_names


def save_preprocessor(preprocessor, filepath):
    """
    儲存前處理器
    
    Args:
        preprocessor: 已訓練的前處理器
        filepath (str or Path): 儲存路徑
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(preprocessor, filepath)
    print(f"✅ 前處理器已儲存到: {filepath}")


def load_preprocessor(filepath):
    """
    載入前處理器
    
    Args:
        filepath (str or Path): 前處理器檔案路徑
        
    Returns:
        ColumnTransformer: 載入的前處理器
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"找不到前處理器檔案: {filepath}")
    
    preprocessor = joblib.load(filepath)
    print(f"✅ 前處理器已載入: {filepath}")
    return preprocessor


def get_feature_names(preprocessor):
    """
    取得處理後的特徵名稱（用於解釋模型）
    
    Args:
        preprocessor: 已 fit 的 ColumnTransformer
        
    Returns:
        list: 特徵名稱列表
    """
    try:
        feature_names = []
        
        # 取得數值型特徵名稱
        if 'num' in preprocessor.named_transformers_:
            num_transformer = preprocessor.named_transformers_['num']
            num_features = preprocessor.transformers_[0][2]  # 數值型欄位名稱
            feature_names.extend(num_features)
        
        # 取得類別型特徵名稱（OneHot 後）
        if 'cat' in preprocessor.named_transformers_:
            cat_transformer = preprocessor.named_transformers_['cat']
            cat_features = preprocessor.transformers_[1][2]  # 類別型欄位名稱
            
            if hasattr(cat_transformer.named_steps['onehot'], 'get_feature_names_out'):
                # 新版本 sklearn
                cat_names = cat_transformer.named_steps['onehot'].get_feature_names_out(cat_features)
            else:
                # 舊版本 sklearn
                cat_names = cat_transformer.named_steps['onehot'].get_feature_names(cat_features)
                
            feature_names.extend(cat_names)
        
        return feature_names
        
    except Exception as e:
        print(f"⚠️ 無法取得特徵名稱: {e}")
        return [f"feature_{i}" for i in range(preprocessor.transform([[0]*len(preprocessor.transformers_[0][2])]).shape[1])]


def get_preprocessing_info(preprocessor):
    """
    顯示前處理器的詳細資訊
    
    Args:
        preprocessor: ColumnTransformer 前處理器
    """
    print("\n📊 前處理器資訊:")
    print("=" * 50)
    
    for name, transformer, columns in preprocessor.transformers:
        print(f"\n🔹 {name.upper()} 轉換器:")
        print(f"   欄位: {columns}")
        print(f"   步驟:")
        for step_name, step_transformer in transformer.steps:
            print(f"     - {step_name}: {step_transformer}")
    
    print(f"\n🔹 其他處理: {preprocessor.remainder}")


def validate_data_for_preprocessing(df, required_columns=None):
    """
    驗證資料是否適合進行前處理
    
    Args:
        df (pd.DataFrame): 輸入資料
        required_columns (list): 必要的欄位列表
        
    Returns:
        tuple: (是否有效, 錯誤訊息)
    """
    if required_columns is None:
        required_columns = ['Age', 'Fare', 'SibSp', 'Parch', 'Sex', 'Embarked', 'Pclass']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        error_msg = f"缺少必要欄位: {missing_columns}"
        return False, error_msg
    
    if df.empty:
        return False, "資料為空"
    
    return True, "資料驗證通過"


if __name__ == "__main__":
    # 測試用程式碼
    print("🧪 測試前處理器...")
    
    # 測試基本版本
    print("\n1. 測試基本前處理器:")
    basic_preprocessor = get_preprocessor(use_advanced_features=False)
    get_preprocessing_info(basic_preprocessor)
    
    # 測試進階版本
    print("\n2. 測試進階前處理器:")
    advanced_preprocessor = get_preprocessor(use_advanced_features=True)
    get_preprocessing_info(advanced_preprocessor)
    
    # 測試特徵建立
    print("\n3. 測試特徵建立:")
    sample_data = pd.DataFrame({
        'Age': [22, 35, np.nan],
        'Fare': [7.25, 71.83, 8.05],
        'SibSp': [1, 1, 0],
        'Parch': [0, 0, 0],
        'Sex': ['male', 'female', 'male'],
        'Embarked': ['S', 'C', 'Q'],
        'Pclass': [3, 1, 3],
        'Name': ['Braund, Mr. Owen Harris', 'Smith, Mrs. John', 'Johnson, Miss. Emily']
    })
    
    enhanced_data = create_basic_features(sample_data)
    print("新增的特徵:")
    new_columns = set(enhanced_data.columns) - set(sample_data.columns)
    for col in new_columns:
        print(f"  - {col}: {enhanced_data[col].tolist()}")
    
    print("\n🎉 前處理器測試完成！")
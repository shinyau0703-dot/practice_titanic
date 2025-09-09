"""
特徵前處理模組
負責建立 ColumnTransformer，處理類別變數編碼、數值變數縮放等
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib


def get_preprocessor():
    """
    建立並回傳前處理管線 (ColumnTransformer)
    
    Returns:
        ColumnTransformer: 完整的前處理管線
    """
    
    # 數值型欄位的處理管線
    numeric_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 類別型欄位的處理管線  
    categorical_features = ['Sex', 'Embarked', 'Pclass', 'Title']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
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


def preprocess_data(df, preprocessor=None, is_training=True):
    """
    使用 ColumnTransformer 前處理資料
    
    Args:
        df (pd.DataFrame): 輸入資料
        preprocessor: 已訓練的前處理器，如果是 None 會建立新的
        is_training (bool): 是否為訓練資料（決定要不要 fit）
        
    Returns:
        tuple: (處理後的特徵矩陣, 前處理器)
    """
    
    if preprocessor is None:
        preprocessor = get_preprocessor()
    
    if is_training:
        # 訓練時：fit_transform
        X_processed = preprocessor.fit_transform(df)
    else:
        # 測試時：只 transform
        X_processed = preprocessor.transform(df)
    
    return X_processed, preprocessor


def save_preprocessor(preprocessor, filepath):
    """儲存前處理器"""
    joblib.dump(preprocessor, filepath)
    print(f"前處理器已儲存到: {filepath}")


def load_preprocessor(filepath):
    """載入前處理器"""
    preprocessor = joblib.load(filepath)
    print(f"前處理器已載入: {filepath}")
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
        # 取得數值型特徵名稱
        numeric_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
        
        # 取得類別型特徵名稱（OneHot 後）
        categorical_features = ['Sex', 'Embarked', 'Pclass', 'Title']
        cat_transformer = preprocessor.named_transformers_['cat']
        
        if hasattr(cat_transformer.named_steps['onehot'], 'get_feature_names_out'):
            cat_names = cat_transformer.named_steps['onehot'].get_feature_names_out(categorical_features)
        else:
            # 舊版本 sklearn
            cat_names = cat_transformer.named_steps['onehot'].get_feature_names(categorical_features)
            
        # 合併所有特徵名稱
        all_features = numeric_features + list(cat_names)
        return all_features
        
    except Exception as e:
        print(f"無法取得特徵名稱: {e}")
        return None


if __name__ == "__main__":
    # 測試用程式碼
    print("測試前處理器建立...")
    preprocessor = get_preprocessor()
    print("✅ 前處理器建立成功！")
    
    # 顯示前處理器的組成（還沒 fit 之前）
    print("\n前處理器組成:")
    for name, transformer, columns in preprocessor.transformers:
        print(f"  {name}: {columns}")
        print(f"    管線步驟: {transformer.steps}")
    
    print("\n🎉 前處理器測試完成！")
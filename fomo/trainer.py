import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

def train_fomo_classifier(data_path="data/fomo_dataset.csv", model_dir="models"):
    """
    從 CSV 檔案載入資料，訓練一個隨機森林分類器，
    用以根據量化與質化指標，預測當前市場的 FOMO 等級。
    """
    print("=== 開始載入訓練數據集 ===")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到數據集：{data_path}。請先執行 collect_data.py！")
        
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"成功載入數據，共有 {len(df)} 筆交易紀錄。")
    
    # 定義特徵欄位與目標欄位
    # 我們使用的是原始指標，沒有經過滾動百分比處理，這樣模型可以學習原始數據在不同歷史時期的相對關係
    feature_cols = [
        'momentum',         # 標普 500 偏離 125MA 程度
        'spec_ratio',       # SPHB / SPLV 投機比值
        'safe_haven_diff',  # SPY vs TLT 收益差
        'junk_bond_diff',   # HYG vs LQD 收益差
        'vix_raw',          # 原始 VIX 波動度
        'sentiment_score'   # 社交與新聞情緒分數 (0-100)
    ]
    target_col = 'fomo_label'
    
    X = df[feature_cols]
    y = df[target_col]
    
    # 進行資料切分 (Train / Test Split)
    # 我們保留最近 20% 的數據作為測試集，以驗證模型對最新市場週期的泛化能力
    # 同時打亂 (shuffle=True) 進行隨機拆分，保證各分類在訓練和測試集都有足夠的樣本
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"訓練集大小: {X_train.shape[0]}，測試集大小: {X_test.shape[0]}")
    
    print("\n=== 建立機器學習流水線 (Pipeline) ===")
    # 我們將 StandardScaler 與 RandomForestClassifier 打包在一個 Pipeline 中
    # 這能有效防止數據洩露 (Data Leakage) 並簡化預測時的調用流程
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=4,
            random_state=42,
            class_weight='balanced' # 處理各類別樣本不完全均衡的情況
        ))
    ])
    
    print("開始訓練隨機森林分類模型...")
    pipeline.fit(X_train, y_train)
    
    print("\n=== 評估模型效能 ===")
    # 預測測試集
    y_pred = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"測試集分類準確率 (Accuracy): {accuracy:.4f}")
    
    # 定義類別名稱對照
    target_names = [
        "0: Extreme Panic (極度恐慌)",
        "1: Fear (恐慌)",
        "2: Neutral (中立)",
        "3: FOMO (貪婪/FOMO)",
        "4: Extreme FOMO (極度貪婪/FOMO)"
    ]
    
    print("\n詳細分類報告 (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # 輸出特徵重要性 (Feature Importances)
    classifier = pipeline.named_steps['classifier']
    importances = classifier.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\n特徵重要性分析 (Feature Importances):")
    for idx, row in feature_importance_df.iterrows():
        print(f"- {row['Feature']}: {row['Importance']:.4f}")
        
    # 保存模型
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model_path = os.path.join(model_dir, "fomo_model.pkl")
    joblib.dump(pipeline, model_path)
    
    print(f"\n=== 模型打包導出成功！ ===")
    print(f"檔案路徑: {model_path}")
    return model_path

if __name__ == "__main__":
    train_fomo_classifier()

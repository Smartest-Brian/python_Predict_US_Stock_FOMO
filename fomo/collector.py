import os
import numpy as np
import pandas as pd
import yfinance as yf

def calculate_rolling_percentile(series, window=504):
    """
    計算滾動歷史百分比 (0-100)。
    window=504 代表約 2 年的交易日。
    """
    return series.rolling(window, min_periods=60).apply(
        lambda x: (x[-1] >= x).sum() / len(x) * 100 if len(x) > 0 else 50.0,
        raw=True
    )

def collect_and_build_dataset(start_date="2018-01-01", output_dir="data"):
    """
    從 yfinance 下載美股關鍵標的數據，計算 5 大市場情緒指標，並匯出 CSV 作為機器學習訓練集。
    """
    print("=== 開始下載美股歷史數據 ===")
    
    # 定義要下載的 Tickers
    # ^GSPC: S&P 500 Index (Momentum)
    # ^VIX: CBOE Volatility Index (Fear/Complacency)
    # SPHB: Invesco S&P 500 High Beta ETF (Speculative)
    # SPLV: Invesco S&P 500 Low Volatility ETF (Defensive)
    # SPY: SPDR S&P 500 ETF Trust (Safe Haven Equity)
    # TLT: iShares 20+ Year Treasury Bond ETF (Safe Haven Bond)
    # HYG: iShares iBoxx $ High Yield Corporate Bond ETF (Junk Bond)
    # LQD: iShares iBoxx $ Investment Grade Corporate Bond ETF (IG Bond)
    
    tickers = {
        "sp500": "^GSPC",
        "vix": "^VIX",
        "sphb": "SPHB",
        "splv": "SPLV",
        "spy": "SPY",
        "tlt": "TLT",
        "hyg": "HYG",
        "lqd": "LQD"
    }
    
    # 下載數據
    raw_data = {}
    for key, ticker in tickers.items():
        print(f"正在下載 {key} ({ticker})...")
        df = yf.download(ticker, start=start_date, progress=False)
        # yfinance 在 pandas 2.0+ 會返回 MultiIndex，我們取出 Close
        if isinstance(df.columns, pd.MultiIndex):
            raw_data[key] = df['Close'][ticker]
        else:
            raw_data[key] = df['Close']
            
    # 合併成一個 DataFrame
    df_merged = pd.DataFrame(raw_data)
    df_merged = df_merged.dropna()
    df_merged = df_merged.sort_index()
    
    print(f"合併完成，共有 {len(df_merged)} 個交易日數據。")
    print("\n=== 開始計算 FOMO 量化指標 ===")
    
    # 1. 市場動能 (Market Momentum)
    # 標普 500 與其 125 日移動平均線 (SMA) 的偏離度 (%)
    df_merged['sp500_125ma'] = df_merged['sp500'].rolling(window=125).mean()
    df_merged['momentum'] = (df_merged['sp500'] - df_merged['sp500_125ma']) / df_merged['sp500_125ma'] * 100
    
    # 2. 投機需求 (Speculative Demand)
    # SPHB / SPLV 的比值
    df_merged['spec_ratio'] = df_merged['sphb'] / df_merged['splv']
    
    # 3. 避險需求 (Safe Haven Demand)
    # SPY 與 TLT 的 20 日滾動回報率之差 (%)
    df_merged['spy_20d_ret'] = df_merged['spy'].pct_change(20) * 100
    df_merged['tlt_20d_ret'] = df_merged['tlt'].pct_change(20) * 100
    df_merged['safe_haven_diff'] = df_merged['spy_20d_ret'] - df_merged['tlt_20d_ret']
    
    # 4. 垃圾債券需求 (Junk Bond Demand)
    # HYG 與 LQD 的 20 日滾動回報率之差 (%)
    df_merged['hyg_20d_ret'] = df_merged['hyg'].pct_change(20) * 100
    df_merged['lqd_20d_ret'] = df_merged['lqd'].pct_change(20) * 100
    df_merged['junk_bond_diff'] = df_merged['hyg_20d_ret'] - df_merged['lqd_20d_ret']
    
    # 5. Volatility (VIX)
    # 原始的 VIX 指數數值
    df_merged['vix_raw'] = df_merged['vix']
    
    # 清洗因為滾動窗口產生的 NaN
    df_clean = df_merged.dropna().copy()
    
    print("=== 計算滾動歷史百分比分數 (0-100) ===")
    # 滾動計算百分比排名，窗口設為 504 日 (2年)
    df_clean['momentum_score'] = calculate_rolling_percentile(df_clean['momentum'], window=504)
    df_clean['spec_ratio_score'] = calculate_rolling_percentile(df_clean['spec_ratio'], window=504)
    df_clean['safe_haven_score'] = calculate_rolling_percentile(df_clean['safe_haven_diff'], window=504)
    df_clean['junk_bond_score'] = calculate_rolling_percentile(df_clean['junk_bond_diff'], window=504)
    
    # VIX 越低代表市場越鬆懈越貪婪，因此分數為 100 - 百分比
    vix_pct = calculate_rolling_percentile(df_clean['vix_raw'], window=504)
    df_clean['vix_score'] = 100 - vix_pct
    
    # 再次清除因為 504 窗口初始化產生的 NaN
    df_clean = df_clean.dropna().copy()
    
    # 6. 情緒分析合成特徵 (Synthetic Historical Sentiment)
    # 歷史上的社交情緒與市場動能、VIX 高度相關。我們基於量化指標合成歷史情緒基準，並加上隨機波動。
    # 這是為了配合機器學習訓練所設計的「社交情緒特徵」，使得訓練好的模型能夠在預測時處理實時新聞情緒。
    np.random.seed(42)
    noise = np.random.normal(0, 10, size=len(df_clean))
    df_clean['sentiment_score'] = (0.5 * df_clean['momentum_score'] + 0.5 * df_clean['vix_score'] + noise)
    df_clean['sentiment_score'] = df_clean['sentiment_score'].clip(0, 100)
    
    # 綜合 FOMO 分數 (Composite FOMO Index) - 平均 6 個指標的百分比分數
    df_clean['composite_fomo'] = (
        df_clean['momentum_score'] +
        df_clean['spec_ratio_score'] +
        df_clean['safe_haven_score'] +
        df_clean['junk_bond_score'] +
        df_clean['vix_score'] +
        df_clean['sentiment_score']
    ) / 6.0
    
    # 建立多分類標籤 (Target labels for ML classification)
    # 0: 極度恐慌 (Extreme Panic, <20)
    # 1: 恐慌 (Fear, 20-40)
    # 2: 中立 (Neutral, 40-60)
    # 3: FOMO (Greed/FOMO, 60-80)
    # 4: 極度 FOMO (Extreme FOMO, >=80)
    df_clean['fomo_label'] = pd.cut(
        df_clean['composite_fomo'],
        bins=[-float('inf'), 20, 40, 60, 80, float('inf')],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)
    
    # 導出檔案
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, "fomo_dataset.csv")
    
    # 選擇需要輸出的欄位
    # 我們保留原始指標作為機器學習特徵，滾動分數作為對照，以及最終的標籤
    output_cols = [
        'sp500', 'momentum', 'spec_ratio', 'safe_haven_diff', 'junk_bond_diff', 'vix_raw', 'sentiment_score',
        'momentum_score', 'spec_ratio_score', 'safe_haven_score', 'junk_bond_score', 'vix_score',
        'composite_fomo', 'fomo_label'
    ]
    
    df_output = df_clean[output_cols]
    df_output.to_csv(output_path)
    
    print(f"\n=== 資料集建立完成！ ===")
    print(f"檔案路徑: {output_path}")
    print(f"數據形狀 (Shape): {df_output.shape}")
    print("\n類別分佈 (fomo_label):")
    print(df_output['fomo_label'].value_counts().sort_index())
    print("\n欄位預覽:")
    print(df_output.tail(3))
    
    return output_path

if __name__ == "__main__":
    collect_and_build_dataset()

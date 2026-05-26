import os
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from fomo.sentiment_analyzer import MarketSentimentAnalyzer

def calculate_percentile(val, series):
    """計算數值在特定數列中的百分位數 (0-100)"""
    if len(series) == 0:
        return 50.0
    return (val >= series).sum() / len(series) * 100

class FomoPredictor:
    def __init__(self, model_path="models/fomo_model.pkl"):
        self.model_path = model_path
        self.pipeline = None
        self.sentiment_analyzer = MarketSentimentAnalyzer()
        
    def load_model(self):
        """載入機器學習模型，若模型不存在則自動拋出異常"""
        if not os.path.exists(self.model_path):
            return False
        try:
            self.pipeline = joblib.load(self.model_path)
            return True
        except Exception as e:
            print(f"載入模型失敗: {str(e)}")
            return False

    def predict_current_fomo(self):
        """
        實時抓取當前市場數據與新聞情緒，計算綜合 FOMO 分數，
        並透過已訓練的機器學習模型進行市場狀態分類預測。
        """
        # 1. 載入模型
        model_loaded = self.load_model()
        
        # 2. 獲取最新社交/新聞情緒
        print("正在分析最新新聞與討論情緒...")
        sentiment_result = self.sentiment_analyzer.get_current_sentiment()
        live_sentiment_score = sentiment_result["overall_score"]
        
        # 3. 獲取最近 500 個交易日的市場數據 (約 2 年) 以計算移動平均與百分位數
        print("正在獲取最新美股指標數據...")
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
        
        raw_data = {}
        for key, ticker in tickers.items():
            # 下載最近 700 天的數據確保扣除掉 NaN 後仍有 504 天以上的窗口
            df = yf.download(ticker, period="3y", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                raw_data[key] = df['Close'][ticker]
            else:
                raw_data[key] = df['Close']
                
        df_merged = pd.DataFrame(raw_data).dropna().sort_index()
        
        # 計算技術指標
        df_merged['sp500_125ma'] = df_merged['sp500'].rolling(window=125).mean()
        df_merged['momentum'] = (df_merged['sp500'] - df_merged['sp500_125ma']) / df_merged['sp500_125ma'] * 100
        df_merged['spec_ratio'] = df_merged['sphb'] / df_merged['splv']
        
        df_merged['spy_20d_ret'] = df_merged['spy'].pct_change(20) * 100
        df_merged['tlt_20d_ret'] = df_merged['tlt'].pct_change(20) * 100
        df_merged['safe_haven_diff'] = df_merged['spy_20d_ret'] - df_merged['tlt_20d_ret']
        
        df_merged['hyg_20d_ret'] = df_merged['hyg'].pct_change(20) * 100
        df_merged['lqd_20d_ret'] = df_merged['lqd'].pct_change(20) * 100
        df_merged['junk_bond_diff'] = df_merged['hyg_20d_ret'] - df_merged['lqd_20d_ret']
        
        df_merged = df_merged.dropna()
        
        # 獲取今日/當前的最新數值
        latest_row = df_merged.iloc[-1]
        
        cur_momentum = latest_row['momentum']
        cur_spec_ratio = latest_row['spec_ratio']
        cur_safe_haven_diff = latest_row['safe_haven_diff']
        cur_junk_bond_diff = latest_row['junk_bond_diff']
        cur_vix = latest_row['vix']
        
        # 計算相對於過去 2 年 (504 天) 的百分比得分 (0-100)
        hist_window = df_merged.iloc[-504:]
        
        momentum_score = calculate_percentile(cur_momentum, hist_window['momentum'])
        spec_ratio_score = calculate_percentile(cur_spec_ratio, hist_window['spec_ratio'])
        safe_haven_score = calculate_percentile(cur_safe_haven_diff, hist_window['safe_haven_diff'])
        junk_bond_score = calculate_percentile(cur_junk_bond_diff, hist_window['junk_bond_diff'])
        vix_score = 100.0 - calculate_percentile(cur_vix, hist_window['vix']) # VIX 反比
        
        # 4. 計算綜合 FOMO 得分 (0-100)
        composite_fomo = (
            momentum_score +
            spec_ratio_score +
            safe_haven_score +
            junk_bond_score +
            vix_score +
            live_sentiment_score
        ) / 6.0
        
        # 5. 執行機器學習分類預測
        features_dict = {
            'momentum': cur_momentum,
            'spec_ratio': cur_spec_ratio,
            'safe_haven_diff': cur_safe_haven_diff,
            'junk_bond_diff': cur_junk_bond_diff,
            'vix_raw': cur_vix,
            'sentiment_score': live_sentiment_score
        }
        
        # 轉換成 Pandas DataFrame 格式，並保證特徵順序與訓練時一致
        features_df = pd.DataFrame([features_dict])
        
        pred_label = 2  # 預設中立
        pred_probs = [0.0, 0.0, 1.0, 0.0, 0.0]
        
        if model_loaded:
            pred_label = int(self.pipeline.predict(features_df)[0])
            pred_probs = self.pipeline.predict_proba(features_df)[0].tolist()
        else:
            # 規則引擎後備機制 (如果模型尚未訓練完成)
            print("⚠️ 警告: 尚未檢測到已訓練的機器學習模型。將採用統計規則引擎進行替代預測。")
            if composite_fomo < 20:
                pred_label = 0
            elif composite_fomo < 40:
                pred_label = 1
            elif composite_fomo < 60:
                pred_label = 2
            elif composite_fomo < 80:
                pred_label = 3
            else:
                pred_label = 4
                
            # 模擬概率
            pred_probs = [0.0] * 5
            pred_probs[pred_label] = 1.0

        # 定義狀態標籤名稱
        status_names = ["Extreme Panic", "Fear", "Neutral", "FOMO", "Extreme FOMO"]
        status_names_zh = ["極度恐慌 (Panic)", "恐慌 (Fear)", "中立 (Neutral)", "貪婪/FOMO", "極度 FOMO (Bubble)"]
        
        predicted_status = status_names[pred_label]
        predicted_status_zh = status_names_zh[pred_label]
        
        # 構建完整結果
        result = {
            "timestamp": df_merged.index[-1].strftime("%Y-%m-%d"),
            "composite_fomo_score": round(composite_fomo, 2),
            "predicted_label": pred_label,
            "predicted_status": predicted_status,
            "predicted_status_zh": predicted_status_zh,
            "model_loaded": model_loaded,
            "prediction_probabilities": {
                status_names[i]: round(prob * 100, 2) for i, prob in enumerate(pred_probs)
            },
            "indicator_breakdown": {
                "momentum": {
                    "raw_value": round(float(cur_momentum), 2),
                    "score": round(momentum_score, 2),
                    "description": "與125日均線偏離度 (越偏上漲代表動能越狂熱)"
                },
                "speculative_demand": {
                    "raw_value": round(float(cur_spec_ratio), 4),
                    "score": round(spec_ratio_score, 2),
                    "description": "高Beta股與低波動股比值 (越高代表市場投機情緒強烈)"
                },
                "safe_haven_demand": {
                    "raw_value": round(float(cur_safe_haven_diff), 2),
                    "score": round(safe_haven_score, 2),
                    "description": "20日股票vs債券回報偏離度 (越高代表資金棄債投股)"
                },
                "junk_bond_demand": {
                    "raw_value": round(float(cur_junk_bond_diff), 2),
                    "score": round(junk_bond_score, 2),
                    "description": "20日高收益債vs投資級債回報偏離度 (越高代表信用風險胃口大開)"
                },
                "market_volatility": {
                    "raw_value": round(float(cur_vix), 2),
                    "score": round(vix_score, 2),
                    "description": "VIX 波動度指數百分比反比 (VIX越低代表市場越鬆懈與樂觀)"
                },
                "social_sentiment": {
                    "raw_value": round(live_sentiment_score, 2),
                    "score": round(live_sentiment_score, 2),
                    "description": "新聞與 Reddit 論壇情緒極性分數 (越高代表言論看多與興奮)"
                }
            },
            "raw_prices": {
                "sp500": round(float(latest_row['sp500']), 2),
                "vix": round(float(latest_row['vix']), 2)
            },
            "sentiment_news": sentiment_result["items"][:10] # 最新 10 條新聞分析詳情
        }
        
        return result

if __name__ == "__main__":
    predictor = FomoPredictor()
    res = predictor.predict_current_fomo()
    print("\n=== 當前美股 FOMO 預測結果 ===")
    print(f"日期: {res['timestamp']}")
    print(f"綜合 FOMO 指數: {res['composite_fomo_score']} / 100")
    print(f"預測市場狀態: {res['predicted_status_zh']}")
    print(f"機器學習分類模型載入狀態: {res['model_loaded']}")
    print("\n分類概率分布:")
    for status, prob in res['prediction_probabilities'].items():
        print(f"- {status}: {prob}%")
    print("\n指標分解得分:")
    for key, data in res['indicator_breakdown'].items():
        print(f"- {key}: {data['score']}分 (原始值: {data['raw_value']})")

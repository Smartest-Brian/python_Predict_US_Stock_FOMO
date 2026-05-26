"""
fomo - 美股 FOMO 指數核心運算套件

本套件提供：
- sentiment_analyzer: 市場新聞與社交情緒分析 (NLTK VADER + RSS)
- collector:          歷史市場數據下載與訓練資料集建構
- trainer:            隨機森林機器學習分類模型訓練與評估
- predictor:          實時市場指標獲取與 FOMO 狀態預測推理
"""

__version__ = "1.0.0"

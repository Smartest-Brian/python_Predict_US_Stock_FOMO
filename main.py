#!/usr/bin/env python3
"""
美股 FOMO 指數與機器學習分類系統 — 主程式進入點

使用方式:
    python main.py              # 啟動 FastAPI 網頁儀表板伺服器
    python -m fomo.collector    # 單獨執行歷史數據採集
    python -m fomo.trainer      # 單獨執行模型訓練
    python -m fomo.predictor    # 單獨執行即時 FOMO 預測
"""

import uvicorn


def main():
    """啟動 FastAPI 網頁伺服器"""
    uvicorn.run(
        "web.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()

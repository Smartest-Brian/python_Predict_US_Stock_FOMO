#!/bin/bash

# ====================================================================
# 美股 FOMO 指數儀表板一鍵啟動腳本 (Auto Start & Launch Script)
# ====================================================================

# 設定文字顏色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo -e "${BLUE}${BOLD}====================================================${NC}"
echo -e "${GREEN}${BOLD}      美股 FOMO 機器學習儀表板 - 一鍵啟動與運行系統${NC}"
echo -e "${BLUE}${BOLD}====================================================${NC}"

# 1. 檢查 Python 虛擬環境
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}⚠️ 未檢測到 .venv 虛擬環境！正在為您初始化環境...${NC}"
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ 建立 .venv 失敗。請檢查是否已在系統安裝 python3。${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ .venv 建立成功！${NC}"
fi

# 2. 激活虛擬環境並檢查依賴
echo -e "${BLUE}正在啟用虛擬環境並載入套件...${NC}"
source .venv/bin/activate

# 3. 檢查套件是否已安裝，如未安裝則安裝
python3 -c "import yfinance, sklearn, fastapi, uvicorn, nltk" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠️ 檢測到尚未安裝完整依賴套件。正在為您自動安裝 requirements.txt...${NC}"
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ 依賴安裝失敗。請檢查您的網路連線。${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ 套件安裝完成！${NC}"
fi

# 4. 檢查模型是否已經存在，如不存在則在啟動前引導訓練
if [ ! -f "models/fomo_model.pkl" ] || [ ! -f "data/fomo_dataset.csv" ]; then
    echo -e "${YELLOW}⚠️ 檢測到系統為首次啟動，且尚未訓練機器學習分類模型。${NC}"
    echo -e "${BLUE}正在自動為您收集歷史數據並訓練模型 (此過程約需 10 秒)...${NC}"
    python3 -m fomo.collector && python3 -m fomo.trainer
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ 模型初始訓練失敗。系統將以統計規則引擎模式啟動。${NC}"
    else
        echo -e "${GREEN}✓ 機器學習分類模型初始化訓練成功！${NC}"
    fi
fi

# 5. 在瀏覽器自動開啟儀表板 (僅適用於 macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    # 延遲 2 秒在背景執行開啟網頁命令，等 FastAPI 伺服器啟動完成
    (sleep 2.5 && open "http://127.0.0.1:8000") &
    echo -e "${GREEN}✓ 已為您預排在伺服器開啟時自動啟動瀏覽器訪問 http://127.0.0.1:8000${NC}"
fi

# 6. 啟動 FastAPI 服務
echo -e "${GREEN}${BOLD}🚀 正在啟動美股 FOMO 儀表板伺服器...${NC}"
echo -e "${YELLOW}提示: 按下 Ctrl + C 可以關閉伺服器並結束運行。${NC}"
echo -e "${BLUE}----------------------------------------------------${NC}"

python3 main.py

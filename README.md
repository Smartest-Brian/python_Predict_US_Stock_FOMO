# 美股 FOMO 指數與機器學習分類儀表板 (US Stock Market FOMO Indicator & ML Dashboard)

本專案是一個全方位的智能美股情緒監控與預測系統，結合了 **5 大量化市場指標** 與 **1 個質化社交/新聞情緒指標**，透過機器學習（隨機森林）分類模型，實時計算出美股當前的 FOMO（Fear Of Missing Out，錯失恐懼症）熱度與市場狀態。

系統提供了一個**霓虹深色毛玻璃風格 (Sleek Dark Mode & Neon Glassmorphism)** 的互動式網頁儀表板，支援歷史趨勢對照圖（標普 500 與歷史當日 FOMO 分數動能染色疊加）、子指標貢獻度雷達與實時情緒新聞。

---

## 📊 情緒指標設計 (The 6 Pillars of FOMO)

系統綜合了以下 6 大維度來評估市場當前的狂熱與恐慌程度：

1. **市場動能 (Market Momentum)**
   - 計算標普 500 指數 (`^GSPC`) 與其 **125 日移動平均線 (SMA)** 的偏離度。偏離度越高，代表買盤速度極快，市場正處於衝刺的 FOMO 狀態。
2. **投機偏好 (Speculative Demand)**
   - 採用**高 Beta 股 ETF (`SPHB`) 與低波動股 ETF (`SPLV`)** 的比值。當散戶與資金瘋狂追逐高風險、高槓桿投機股並拋棄防禦性股時，此比值會迅速攀升。
3. **避險需求 (Safe Haven Demand)**
   - 衡量股票與避險國債的資金分配權衡。我們使用 **S&P 500 (`SPY`) 與 20年+長期美債 (`TLT`)** 的 20 日滾動收益偏離。比值越高，代表資金大舉流向股市，無懼利率風險。
4. **垃圾債券需求 (Junk Bond Demand)**
   - 衡量市場對信用風險的承受力。計算 **高收益債 ETF (`HYG`) 與投資級債 ETF (`LQD`)** 的 20 日滾動收益差。當市場無懼信用違約風險、追逐垃圾債的高回報時，顯示 FOMO 情緒強烈。
5. **波動度懈怠百分比 (VIX Complacency)**
   - Volatility Index (`^VIX`) 處於歷史滾動分布的反比。當 VIX 越接近歷史低點，代表市場極度懈怠、缺乏防備，是 FOMO 買盤的最佳寫照。
6. **社交與新聞情緒 (Social Sentiment)**
   - 免 API Key，透過抓取 Yahoo Finance 的標普新聞 RSS，以及 Reddit 金融論壇 (r/stocks, r/investing) 的最新 RSS 討論，利用 `NLTK VADER` 情緒分析模型，實時計算言論看多極性與興奮度。

---

## 🛠️ 專案目錄結構與模組設計

```
USStockFomo/
├── main.py                          # 主程式進入點 (thin entry point)
├── requirements.txt                 # Python 依賴套件
├── start.sh                         # 一鍵啟動腳本 (macOS 自動開啟瀏覽器)
├── .gitignore                       # Git 版本控制忽略規則
│
├── fomo/                            # 核心商業邏輯套件 (Core Package)
│   ├── __init__.py
│   ├── sentiment_analyzer.py        # RSS 新聞/論壇情緒分析 (NLTK VADER)
│   ├── collector.py                 # 歷史數據下載與訓練資料集建構
│   ├── trainer.py                   # 隨機森林 ML 分類模型訓練與評估
│   └── predictor.py                 # 實時指標獲取與 FOMO 狀態預測推理
│
├── web/                             # 網頁服務層套件 (Web Layer Package)
│   ├── __init__.py
│   ├── app.py                       # FastAPI 路由、API 端點與背景任務
│   ├── static/css/styles.css        # 霓虹暗色調毛玻璃 CSS 樣式表
│   └── templates/index.html         # SVG 儀表盤 + Chart.js 染色歷史圖前端
│
├── docs/                            # 文件與開發指引
│   └── project_prompt.md            # AI 代理提示詞與技術規格手冊
│
├── data/                            # 運行時產生的訓練資料 (gitignored)
│   └── fomo_dataset.csv
├── models/                          # 訓練完成的模型檔案 (gitignored)
│   └── fomo_model.pkl
└── logs/                            # 應用程式日誌 (gitignored)
```

---

## 🚀 快速開始使用

### 方法一：一鍵啟動（推薦）
```bash
./start.sh
```
腳本會自動完成環境檢查、依賴安裝、首次數據下載與模型訓練，並在 macOS 上自動開啟瀏覽器。

### 方法二：手動啟動
```bash
# 1. 啟用虛擬環境
source .venv/bin/activate

# 2. 啟動 FastAPI 儀表板伺服器
python main.py
```
👉 **開啟瀏覽器訪問: [http://127.0.0.1:8000](http://127.0.0.1:8000)**

### 獨立模組執行
```bash
python -m fomo.collector    # 單獨下載歷史數據並產生 CSV
python -m fomo.trainer      # 單獨訓練隨機森林分類模型
python -m fomo.predictor    # 單獨執行即時 FOMO 預測
```

### 一鍵模型重訓
*   系統提供自動統計規則後備機制。若您第一次啟動時尚未生成模型，您可以直接在網頁右上角點擊 **「重新訓練模型」 (Retrain Model)** 按鈕。
*   系統將在背景自動下載 5 年歷史數據並重新訓練。訓練完成後，網頁將自動重整載入最新的隨機森林機器學習預測！

---

## 🔮 機器學習分類狀態說明

預測出來的市場狀態可分為 5 個等級，並在儀表板中以不同霓虹色彩呈現：

| 狀態名稱 (Status) | 評分範圍 (composite_fomo) | 視覺色彩 | 意涵 |
| :--- | :---: | :---: | :--- |
| **極度恐慌 (Extreme Panic)** | `< 20` | 🔴 霓虹深紅 | 市場極度恐慌，拋售狂潮，VIX 飆升，此時往往是歷史買點（Fear peak）。 |
| **恐慌 (Fear)** | `20 - 40` | 🟠 霓虹橘 | 避險需求上升，資金流向國債，情緒悲觀。 |
| **中立 (Neutral)** | `40 - 60` | 🔵 科技冰藍 | 多空交織，各指標處於歷史平均水平。 |
| **貪婪 / FOMO (FOMO)** | `60 - 80` | 🟢 極光亮綠 | 投機偏好大開，買盤動能猛烈，論壇一片看多。 |
| **極度 FOMO (Extreme FOMO)** | `>= 80` | 🟡 炫目亮金 | speculative bubble 階段，市場極度懈怠，資金大舉槓桿湧入投機股，警惕短線見頂。 |

# 美股 FOMO 指數專案開發與 AI 代理提示詞手冊 (Project Prompt & Developer Manual)

本文件是專為開發人員與未來引進的 AI 程式設計代理（如 Antigravity, Claude, ChatGPT）設計的**專案維護與擴展指南**。它詳細說明了系統的核心邏輯、數學模型、機器學習特徵，並提供了一套即插即用的 AI 提示詞模板。

---

## 🧭 專案全景與開發邏輯

本專案名為 **美股 FOMO 指數與機器學習分類系統 (US Stock Market FOMO Index)**。其核心目的是利用金融量化指標與即時社交討論，客觀評估目前美國股市的熱度、狂熱（FOMO）程度與恐慌（Panic）程度。

### 1. 核心指標數學公式與處理 (The 6 Pillars)

為了在將來擴展指標或進行除錯，請遵循以下數學與數據處理規範：

1.  **市場動能 (Momentum, `momentum`)**：
    $$\text{Momentum}_t = \frac{\text{SP500}_t - \text{SMA}_{125}(\text{SP500})_t}{\text{SMA}_{125}(\text{SP500})_t} \times 100$$
    *說明*：偏離 125 日均線的百分比。越高代表上漲動能越極端。

2.  **投機偏好 (Speculative Demand, `spec_ratio`)**：
    $$\text{Speculate Ratio}_t = \frac{\text{SPHB}_t}{\text{SPLV}_t}$$
    *說明*：標普高 Beta ETF 價格除以標普低波動 ETF 價格。越高代表投機資金大量湧入風險資產。

3.  **避險需求偏離 (Safe Haven Demand, `safe_haven_diff`)**：
    $$\text{Safe Haven Diff}_t = (\text{SPY}_t \text{ 20日報酬率} - \text{TLT}_t \text{ 20日報酬率}) \times 100$$
    *說明*：標普 500 ETF 報酬率減去長期美債 ETF 報酬率。反映資金流向股市（FOMO）或流向避險國債（Fear）。

4.  **垃圾債券需求偏離 (Junk Bond Demand, `junk_bond_diff`)**：
    $$\text{Junk Bond Diff}_t = (\text{HYG}_t \text{ 20日報酬率} - \text{LQD}_t \text{ 20日報酬率}) \times 100$$
    *說明*：高收益債 ETF 報酬率減去投資級債 ETF 報酬率。反映信用風險溢價胃口。

5.  **波動度指數百分位反比 (VIX Complacency, `vix_score`)**：
    $$\text{VIX Score}_t = 100 - \text{Rolling Percentile}_{504}(\text{VIX}_t)$$
    *說明*：VIX 在歷史 2 年（504 日）滾動分佈的百分位反比。VIX 越低，Complacency 越高，FOMO 分數越接近 100。

6.  **社交新聞情緒 (Social Sentiment, `sentiment_score`)**：
    $$\text{Sentiment Score}_t = (\text{VADER Compound Score} + 1.0) \times 50.0$$
    *說明*：抓取新聞與討論標題，經 VADER 情緒分類後歸一化至 `0 - 100`。

---

### 2. 歷史百分比排名 (Rolling Percentile Normalization)

所有指標的滾動歷史百分比排名皆基於 **504 個交易日** (約 2 年滾動窗口) 來進行計算，以排除長週期的結構性市場漂移：

```python
def calculate_rolling_percentile(series, window=504):
    return series.rolling(window, min_periods=60).apply(
        lambda x: (x[-1] >= x).sum() / len(x) * 100 if len(x) > 0 else 50.0,
        raw=True
    )
```

---

### 3. 機器學習模型架構

- **特徵數 (6 個)**: `['momentum', 'spec_ratio', 'safe_haven_diff', 'junk_bond_diff', 'vix_raw', 'sentiment_score']`
- **分類模型**: `RandomForestClassifier(n_estimators=150, max_depth=10, class_weight='balanced')`
- **持久化**: 模型與 `StandardScaler` 以隨機森林 Pipeline 打包，導出至 `models/fomo_model.pkl`。
- **狀態標籤與評分範圍**:
  - `0 - 20`: **極度恐慌 (Extreme Panic)**
  - `20 - 40`: **恐慌 (Fear)**
  - `40 - 60`: **中立 (Neutral)**
  - `60 - 80`: **貪婪 / FOMO (FOMO)**
  - `80 - 100`: **極度 FOMO (Extreme FOMO)**

---

## 🤖 專屬 AI 提示詞模板 (System Prompts for AI Agents)

如果您在未來需要委託 AI 助手對本專案進行修改，可以直接**複製並貼上**以下對應的提示詞：

### 提示詞 1：為專案新增新的技術指標（例如：Put/Call 比例、融資餘額）
> [!TIP]
> **複製下方提示詞給 AI 助手：**
> ```text
> 我有一個「美股 FOMO 指數與機器學習分類系統」專案，採用 Python 套件化目錄結構：核心邏輯在 `fomo/` 套件，Web 層在 `web/` 套件。
> 我想在 `fomo/collector.py` 和 `fomo/predictor.py` 中新增一個量化指標：「Put/Call 期權比率」或其代理指標（例如 CBOE 的期權交易量比例，或利用 yfinance 抓取指數 ETF 的 Options 交易特徵）。
> 
> 請幫我：
> 1. 修改 `fomo/collector.py` 中的 `collect_and_build_dataset` 方法，下載該指標的歷史數據。
> 2. 將該指標以 504 日滾動窗口計算滾動百分比（注意如果是恐慌指標要計算 100 - 百分比）。
> 3. 更新 `fomo/trainer.py` 的特徵欄位，並將此新特徵納入隨機森林的 Pipeline 訓練中。
> 4. 修改 `fomo/predictor.py`，在實時預測中加入對該指標的即時下載與百分比計算。
> 5. 修改 `web/templates/index.html`，在 6 大子指標網頁卡片中擴展為 7 大指標卡片，展示新指標的得分。
> 請確保修改符合現有代碼的結構與變數命名規範（fomo/ 套件內的模組使用 `from fomo.xxx import` 語法）。
> ```

### 提示詞 2：網頁儀表板 UI 美化、新增圖表與中英文切換
> [!TIP]
> **複製下方提示詞給 AI 助手：**
> ```text
> 我有一個美股 FOMO 指數監控儀表板專案，後端是 FastAPI（路由定義在 `web/app.py`），前端是 HTML + CSS + Chart.js（位於 `web/templates/` 與 `web/static/`）。
> 我想要美化前端儀表板，並加入以下功能：
> 1. 在左側 FOMO 圓環儀表盤下方，新增一個中英文切換開關 (Language Switcher)，點擊可實時將整個網頁切換為繁體中文或英文。
> 2. 將 Chart.js 歷史標普 500 折線染色疊加圖，新增一個「顯示/隱藏 FOMO 染色折線」的按鈕，點擊可切換成單色線條或 FOMO 分色染色。
> 3. 為 live 新聞情绪列表新增一個分頁功能 (Pagination) 或「顯示更多」按鈕，限制初始只顯示 5 條，點擊展示更多，且添加滑動淡入動畫。
> 請幫我修改 `web/templates/index.html` 以及 `web/static/css/styles.css`，保持毛玻璃與霓虹暗色調的 premium 視覺質感。
> ```

### 提示詞 3：除錯與日誌系統優化
> [!TIP]
> **複製下方提示詞給 AI 助手：**
> ```text
> 我的美股 FOMO 預測系統中，當 RSS 情緒分析（例如 Yahoo Finance 新聞或 Reddit API）因為請求過於頻繁遭遇 HTTP 429 錯誤時，雖然現有代碼有 try-except 不會崩潰，但我想在系統中建立一個更完善的日誌紀錄 (Logging)。
> 
> 請幫我：
> 1. 在 `fomo/sentiment_analyzer.py`, `fomo/predictor.py` 中引入 Python 內建的 `logging` 模組，並在專案根目錄輸出 `logs/fomo_app.log` 檔案。
> 2. 格式化日誌輸出，包括：時間戳、模組名稱、日誌級別、日誌內容。
> 3. 確保 `web/app.py` 的 FastAPI 後端也會將請求日誌與背景重訓日誌寫入該 log 檔，並在重訓失敗時捕獲完整的 Traceback 寫入日誌。
> ```

---

## 🛠️ 開發維護命令行速查表 (CLI Reference)

在專案目錄下進行維護時的常用命令：

| 任務目標 | 執行命令 | 預期輸出 |
| :--- | :--- | :--- |
| **激活虛擬環境** | `source .venv/bin/activate` | 終端機前綴出現 `(.venv)` |
| **一鍵啟動** | `./start.sh` | 自動檢查環境、訓練模型、啟動伺服器與瀏覽器 |
| **手動採集數據** | `python -m fomo.collector` | 生成 `data/fomo_dataset.csv` |
| **手動訓練模型** | `python -m fomo.trainer` | 輸出 90%+ 準確率報告與 `models/fomo_model.pkl` |
| **手動測試預測** | `python -m fomo.predictor` | 終端輸出當天預測狀態與機率 JSON 分佈 |
| **啟動 Web 服務** | `python main.py` | 啟動本地 `http://127.0.0.1:8000` 網頁服務 |

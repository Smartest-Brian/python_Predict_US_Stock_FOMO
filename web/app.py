import os
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import pandas as pd
from fomo.predictor import FomoPredictor
from fomo.collector import collect_and_build_dataset
from fomo.trainer import train_fomo_classifier

# 解析本檔案所在目錄，用於定位 static/ 與 templates/ 的絕對路徑
_WEB_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(
    title="美股 FOMO 指標與機器學習分類系統",
    description="實時評估美股整體 FOMO 程度的機器學習儀表板",
    version="1.0.0"
)

# 掛載靜態資源與模板（使用絕對路徑，確保無論 CWD 為何都能正確定位）
app.mount("/static", StaticFiles(directory=os.path.join(_WEB_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(_WEB_DIR, "templates"))

# 初始化預測器
predictor = FomoPredictor()

# 全域狀態，用以顯示目前背景是否正在訓練模型
training_status = {
    "is_training": False,
    "last_trained": "未進行過訓練",
    "message": "系統就緒"
}

def bg_retrain_task():
    """在背景執行數據收集與重新訓練的執行緒任務"""
    global training_status
    try:
        training_status["is_training"] = True
        training_status["message"] = "正在下載最新歷史市場數據..."
        
        # 1. 採集最新數據
        collect_and_build_dataset()
        
        training_status["message"] = "正在重新訓練機器學習模型..."
        # 2. 重新訓練模型
        train_fomo_classifier()
        
        # 3. 重新載入模型
        predictor.load_model()
        
        training_status["last_trained"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        training_status["message"] = "訓練成功完成！"
    except Exception as e:
        training_status["message"] = f"訓練失敗: {str(e)}"
    finally:
        training_status["is_training"] = False

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """渲染儀表板首頁"""
    return templates.TemplateResponse(request=request, name="index.html", context={})

@app.get("/api/fomo")
async def get_fomo_prediction():
    """獲取實時 FOMO 預測結果與指標分解"""
    try:
        res = predictor.predict_current_fomo()
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"實時預測失敗: {str(e)}")

@app.get("/api/history")
async def get_history_data():
    """獲取歷史 S&P 500 與 FOMO 指標對照，供前端 Chart.js 繪製歷史趨勢線"""
    data_path = "data/fomo_dataset.csv"
    if not os.path.exists(data_path):
        # 如果沒有歷史數據，返回空列表
        return []
        
    try:
        # 載入歷史數據
        df = pd.read_csv(data_path, index_col=0)
        
        # 我們只返回最近的 300 個交易日（約 1.2 年的數據），以保證前端渲染流暢
        df_sub = df.tail(300)
        
        history_list = []
        for idx, row in df_sub.iterrows():
            history_list.append({
                "date": str(idx)[:10],
                "sp500": round(float(row["sp500"]), 2),
                "composite_fomo": round(float(row["composite_fomo"]), 2),
                "label": int(row["fomo_label"])
            })
            
        return history_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取歷史數據失敗: {str(e)}")

@app.post("/api/train")
async def trigger_retrain(background_tasks: BackgroundTasks):
    """一鍵觸發背景重新訓練"""
    global training_status
    if training_status["is_training"]:
        return {"status": "running", "message": "模型訓練已在背景執行中，請勿重複提交。"}
        
    training_status["is_training"] = True
    training_status["message"] = "已排程重新訓練任務..."
    
    background_tasks.add_task(bg_retrain_task)
    return {"status": "scheduled", "message": "已成功在背景啟動資料下載與重新訓練。"}

@app.get("/api/train/status")
async def get_train_status():
    """獲取目前模型的訓練狀態"""
    return training_status

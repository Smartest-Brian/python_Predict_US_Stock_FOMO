import re
import xml.etree.ElementTree as ET
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure VADER lexicon is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

class MarketSentimentAnalyzer:
    """
    市場情緒分析器：透過 RSS 訂閱源抓取美股相關新聞與論壇討論，
    並使用 NLTK VADER 模型計算當前市場的情緒分數（0-100）。
    """
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        # 美股相關的 RSS 訂閱源
        self.feeds = {
            "Yahoo Finance (S&P 500)": "https://finance.yahoo.com/rss/headline?s=^GSPC",
            "CNBC Finance News": "https://www.cnbc.com/id/10000664/device/rss/rss.html",
            "Reddit r/stocks": "https://www.reddit.com/r/stocks/.rss",
            "Reddit r/investing": "https://www.reddit.com/r/investing/.rss"
        }

    def _clean_html(self, text):
        """移除文本中的 HTML 標記與多餘空白"""
        if not text:
            return ""
        # 移除 HTML 標記
        clean = re.compile('<.*?>')
        text = re.sub(clean, '', text)
        # 移除多餘空白
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def fetch_feed_items(self, source_name, url):
        """抓取並解析單個 RSS 訂閱源"""
        items = []
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code != 200:
                print(f"[{source_name}] 獲取失敗，HTTP 狀態碼: {response.status_code}")
                return items

            # 解析 XML
            root = ET.fromstring(response.content)
            
            # 處理 RSS 2.0 (channel/item) 或 Atom (feed/entry)
            # RSS 2.0 格式
            channel = root.find('channel')
            if channel is not None:
                for item in channel.findall('item'):
                    title_elem = item.find('title')
                    desc_elem = item.find('description')
                    date_elem = item.find('pubDate') or item.find('{http://purl.org/dc/elements/1.1/}date')
                    link_elem = item.find('link')

                    title = self._clean_html(title_elem.text if title_elem is not None else "")
                    desc = self._clean_html(desc_elem.text if desc_elem is not None else "")
                    date_str = date_elem.text if date_elem is not None else ""
                    link = link_elem.text if link_elem is not None else ""

                    if title:
                        items.append({
                            "title": title,
                            "description": desc,
                            "source": source_name,
                            "pub_date": date_str,
                            "link": link
                        })
            else:
                # Atom 格式 (常見於 Reddit RSS)
                entries = root.findall('{http://www.w3.org/2005/Atom}entry')
                for entry in entries:
                    title_elem = entry.find('{http://www.w3.org/2005/Atom}title')
                    content_elem = entry.find('{http://www.w3.org/2005/Atom}content') or entry.find('{http://www.w3.org/2005/Atom}summary')
                    date_elem = entry.find('{http://www.w3.org/2005/Atom}published') or entry.find('{http://www.w3.org/2005/Atom}updated')
                    link_elem = entry.find('{http://www.w3.org/2005/Atom}link')

                    title = self._clean_html(title_elem.text if title_elem is not None else "")
                    
                    # Reddit 的 content 通常含有大量 HTML，我們需要抓取純文字
                    content_raw = content_elem.text if content_elem is not None else ""
                    if content_raw:
                        soup = BeautifulSoup(content_raw, "html.parser")
                        desc = self._clean_html(soup.get_text())
                    else:
                        desc = ""

                    date_str = date_elem.text if date_elem is not None else ""
                    link = link_elem.attrib.get('href', '') if link_elem is not None else ""

                    if title:
                        items.append({
                            "title": title,
                            "description": desc[:200] + "..." if len(desc) > 200 else desc,
                            "source": source_name,
                            "pub_date": date_str,
                            "link": link
                        })
        except Exception as e:
            print(f"[{source_name}] 解析失敗: {str(e)}")
        
        return items

    def analyze_sentiment(self, text):
        """計算文本的情緒分數（將 VADER 的 compound -1~1 映射至 0~100）"""
        if not text:
            return 50.0, 0.0
        
        scores = self.sia.polarity_scores(text)
        compound = scores['compound']
        
        # 歸一化分數： (compound + 1) * 50 
        # 例如 compound=0.8 => 90 (極度看多/FOMO)
        # compound=-0.5 => 25 (恐慌)
        normalized_score = (compound + 1.0) * 50.0
        return normalized_score, compound

    def get_current_sentiment(self):
        """
        獲取最新所有訂閱源的綜合情緒分析
        """
        all_items = []
        for name, url in self.feeds.items():
            feed_items = self.fetch_feed_items(name, url)
            all_items.extend(feed_items)

        if not all_items:
            print("無法獲取任何新聞與論壇內容，返回預設中立情緒。")
            return {
                "overall_score": 50.0,
                "overall_status": "Neutral",
                "items": []
            }

        scored_items = []
        total_score = 0.0

        for item in all_items:
            # 結合標題與內容進行情緒分析
            text_to_analyze = f"{item['title']}. {item['description']}"
            score, compound = self.analyze_sentiment(text_to_analyze)
            
            # 分類狀態
            if compound >= 0.05:
                status = "Bullish"
            elif compound <= -0.05:
                status = "Bearish"
            else:
                status = "Neutral"

            item_scored = {
                **item,
                "sentiment_score": round(score, 2),
                "compound": round(compound, 4),
                "status": status
            }
            scored_items.append(item_scored)
            total_score += score

        overall_score = total_score / len(scored_items)
        
        if overall_score >= 60.0:
            overall_status = "FOMO"
        elif overall_score <= 40.0:
            overall_status = "Fear"
        else:
            overall_status = "Neutral"

        # 按發布時間或得分排序（此處按 compound 絕對值排序，展示情緒強烈的內容）
        scored_items.sort(key=lambda x: abs(x['compound']), reverse=True)

        return {
            "overall_score": round(overall_score, 2),
            "overall_status": overall_status,
            "items": scored_items[:30] # 僅返回情緒最顯著的前 30 筆
        }

if __name__ == "__main__":
    # 簡單單體測試
    analyzer = MarketSentimentAnalyzer()
    print("正在獲取最新市場情緒...")
    result = analyzer.get_current_sentiment()
    print(f"綜合情緒得分: {result['overall_score']} ({result['overall_status']})")
    print("\n前三條高情緒新聞/討論:")
    for idx, item in enumerate(result['items'][:3]):
        print(f"{idx+1}. [{item['source']}] ({item['status']} - {item['sentiment_score']}): {item['title']}")

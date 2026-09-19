from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment, DataFeed
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

def download_paper_data(symbol):
    print(f"Downloading {symbol}...")
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=pd.Timestamp("2018-01-01", tz="America/New_York"),
        end=pd.Timestamp("2023-09-30", tz="America/New_York"),
        adjustment=Adjustment.SPLIT,
        feed=DataFeed.IEX,
    )
    bars = client.get_stock_bars(request_params)
    df = bars.df.reset_index(level=0, drop=True)
    df.index = df.index.tz_convert("America/New_York")
    df = df.between_time("09:30", "15:59")
    df.to_parquet(f"data/{symbol}_1min_paper.parquet")
    print(f"Saved {len(df)} rows for {symbol}.")
    print(df.head(3))

for s in ["QQQ", "TQQQ"]:
    download_paper_data(s)
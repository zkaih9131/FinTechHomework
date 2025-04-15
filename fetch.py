from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timezone, timedelta
from urllib.request import urlopen
import json
import pandas as pd
import time

def fetch_gnews(api_key: str, query: str, date_str: str):
    """爬取一天的所有相关新闻"""
    start_date_iso = (datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                      .isoformat().replace("+00:00", "Z"))
    end_date_iso = ((datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).replace(tzinfo=timezone.utc)
                    .isoformat().replace("+00:00", "Z"))

    url = (f"https://gnews.io/api/v4/search?q={query}&lang=en&country=us&max=10&from={start_date_iso}"
           f"&to={end_date_iso}&sortby=relevance&apikey={api_key}")

    with urlopen(url) as response:
        data = json.loads(response.read().decode("utf-8"), encoding="utf-8")
        articles = data.get("articles", [])

    return {date_str: articles}


def fetch_news_over_time(api_key: str, query: str, start_date: str, end_date: str, output_file: str, count_date: int):
    """爬取数据，从start_date倒着爬，直到end_date停止"""
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    # 读取已有数据
    try:
        with open(output_file, "r", encoding='utf-8') as f:
            master_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        master_data = {}
    # 爬取数据
    count = 0
    while current_date >= end_date and count < count_date:
        date_str = current_date.strftime("%Y-%m-%d")
        if date_str not in master_data:
            print(f"Fetching news for {date_str}...")
            master_data.update(fetch_gnews(api_key=api_key, query=query, date_str=date_str))
            # 保存date_str当天的数据
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(master_data, f, ensure_ascii=False, indent=4)
            # todo: 待检查是否需要这种方法
            time.sleep(1)  # 这里稍微停一下，防止开始计费
            count = count + 1
            print(f"{count}/{count_date}")
        else:
            print(f"Skipping {date_str}, already fetched.")

        current_date -= timedelta(days=1)

    print("saved: gnews_master.json")


def merge_json_file(files: list, output_file: str):
    """合并多个json文件的数据"""
    # 这段是用来合并两个数据的，理论上需要爬至少200天以上的数据，因为还要切分训练集和测试集
    merged = dict()
    all_dates = set()
    data_list = list()
    # 读取文件
    for file in files:
        with open(file, 'r', encoding='utf-8') as f_in:
            data = json.load(f_in)
            all_dates.add(data.keys())
            data_list.append(data)
    # 处理合并数据
    for date in all_dates:
        for data in data_list:
            merged[date] += data.get(date, [])

    merged_sorted = dict(sorted(merged.items(), key=lambda x: x[0], reverse=True))
    # 输出合并json文件
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(merged_sorted, f_out, ensure_ascii=False, indent=4)

def fetch_price(api_key: str, ticker: str ):
    """爬股票数据"""
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=ticker, outputsize="full")

    data.to_csv(f"./data/{ticker}_historical.csv")

    print("fetch stock saved:", data.head())

def fetch_stock(ticker: str):
    # df = pd.read_csv("AAPL_historical.csv")
    # df["date"] = pd.to_datetime(df["date"])
    # df = df.sort_values("date")
    # df["trend"] = df["4. close"].diff().apply(lambda x: "increase" if x > 0 else ("decrease" if x < 0 else "stable"))
    # df[["date", "4. close", "trend"]].to_csv("AAPL_trend.csv", index=False)
    # 上面这一段是用绝对值的大小判断涨跌，下面这一段会以0.7%为界限判断稳定，应该是下面这种合理，但是测下来这个趋势判断对预测价格的训练没什么用
    # 里面用到的文件名称自己改
    df = pd.read_csv(f"./data/{ticker}_historical.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    threshold = 0.007
    df["pct_change"] = df["4. close"].pct_change() # 百分比变化
    df["trend"] = df["pct_change"].apply(
        lambda x: "increase" if x > threshold else ("decrease" if x < -threshold else "stable")
    )
    df.rename(columns={"4. close": "closingValue"}, inplace=True)
    df[["date", "closingValue", "trend"]].to_csv(f"./data/{ticker}_trend.csv", index=False)
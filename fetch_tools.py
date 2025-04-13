import json
from urllib.request import urlopen
from datetime import datetime, timezone, timedelta
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


def fetch_news_over_time(api_key: str, query: str, start_date: str, end_date: str,
                         output_file: str = "gnews_master_nvd.json"):
    """爬取数据，从start_date倒着爬，直到end_date停止"""
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    # 读取已有数据
    try:
        with open(output_file, "r") as f:
            master_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        master_data = {}
    # 爬取数据
    while current_date >= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        if date_str not in master_data:
            print(f"Fetching news for {date_str}...")
            master_data.update(fetch_gnews(api_key=api_key, query=query, date_str=date_str))
            # 保存date_str当天的数据
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(master_data, f, ensure_ascii=False, indent=4)
            # todo: 待检查是否需要这种方法
            time.sleep(1)  # 这里稍微停一下，防止开始计费
        else:
            print(f"Skipping {date_str}, already fetched.")

        current_date -= timedelta(days=1)

    print("saved: gnews_master.json")


def merge_json_file(*files: str, output_file: str="merged_file.json"):
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

    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(merged_sorted, f_out, ensure_ascii=False, indent=4)

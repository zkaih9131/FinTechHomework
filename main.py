from fetch_tools import *
def main():
    # 0、初始化参数
    api_key = "__自己注册__"  # 这里的API要自己去上面的网址注册一个，免费版一天只能爬100天的新闻:1d754b596757088d38ea69b73ae145c2
    query = "NVIDIA"  # 这是搜索新闻的关键字
    start_date = "2024-12-31"  # 自己算一下日期，别超过100天
    end_date = "2024-09-23"  # 最好是爬2024年8月1日到2025年4月8日的，苹果的股票我用的这个区间
    # 1、爬取新闻数据
    fetch_news_over_time(api_key, query, start_date, end_date)

if __name__ == '__main__':
    main()
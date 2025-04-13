# 爬虫部分参数
api_key = "__自己注册__"  # 这里的API要自己去上面的网址注册一个，免费版一天只能爬100天的新闻:1d754b596757088d38ea69b73ae145c2
query = "NVIDIA"  # 这是搜索新闻的关键字
start_date = "2024-12-31"  # 自己算一下日期，别超过100天
end_date = "2024-09-23"  # 最好是爬2024年8月1日到2025年4月8日的，苹果的股票我用的这个区间
fetch_file = "./data/fetch_data.json" # 爬取数据保存的文件

# 合并json文件
merge_input_files = list(fetch_file) # 需要合并的所有文件，目前仅使用直接输出的数据，不进行合并处理
merge_output_file = f"./data/merged.json"

# 给新闻打分
news_data_file = merge_output_file
saved_score_file = "./data/sentiment_scores_title.csv"

# 爬股票数据
API_KEY = "SSIG9IPUQPWY332H" #这个API好像没有用量限制
ticker = "NVDA"  #这里是在获取股价信息，AAPL是苹果，去查一下你想做的公司的美股代码

# 训练
random_state = 42
sweep_config = {
        "method": "bayes",
        "metric": {"name": "test_mse", "goal": "minimize"},
        "parameters": {
            "hidden_size": {"values": [32, 64]},
            "num_layers": {"values": [1, 2, 3]},
            "lr": {"values": [0.001, 0.0005]},
            "num_epochs": {"value": 3000}
        }
    }

# 测试
# 自己改下路径，加载刚才loss最小的模型
best_model_path = "best_model_h64_l2_lr0.001.pth"

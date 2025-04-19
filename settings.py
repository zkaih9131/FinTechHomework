# 爬虫部分参数
api_keys = [
            # "66d7a16fd18e5ebfb66382503c74c3bd", # mine
            # "3300cfeabb6630b55397a8332bb37a52",
            "f45a867c1b371b63eec8aeade2b3566e",
            "4af60bf83d65e70313def01838550e82",
            "fce4fa8ec34a65b3170cb655b9e89b93",
            "e8abd28075af21d5166a526aa202974d",
            "ae1891813bb7df9694a1ac6193aaa8a7",
            "5be650b7c921ff3c9ad89b210979e192",
            "a3d345be7ad6ea0dad702d3b4269f07a",
            "7291159c92d6e327816a61132913dd4e",
            "3e338c1d9e80f7a90469c21ebcb82aaa",
            "1d132e2daa17378ba8afe2de97a2cd3c",
            "e8750565e5ca191d34c11560cb3406f3",
            "975c4c0f963cf62b60b2b9ed29c341b5"]  # 这里的API要自己去上面的网址注册一个，免费版一天只能爬100天的新闻:1d754b596757088d38ea69b73ae145c2
query = "Microsoft"  # 这是搜索新闻的关键字
start_date = "2025-04-08"  # 自己算一下日期，别超过100天
end_date = "2020-09-06"  # 最好是爬2024年8月1日到2025年4月8日的，苹果的股票我用的这个区间
count_date = 100
fetch_file = "data/microsoft20211127-20250408.json"  # 爬取数据保存的文件

# 合并json文件
merge_input_files = list(fetch_file) # 需要合并的所有文件，目前仅使用直接输出的数据，不进行合并处理
merge_output_file = f"./data/merged.json"

# 给新闻打分
news_data_file = merge_output_file
saved_score_file = "./data/sentiment_scores_title.csv"

# 爬股票数据
API_KEY = "SSIG9IPUQPWY332H" #这个API好像没有用量限制
ticker = "MSFT"  #这里是在获取股价信息，AAPL是苹果，去查一下你想做的公司的美股代码

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

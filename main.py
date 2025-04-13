import wandb

from fetch import *
from test import *
from train import train as training


def fetch():
    """1、爬取新闻数据并进行评分处理"""
    # 1.1、爬取新闻数据
    fetch_news_over_time(api_key, query, start_date, end_date, fetch_file)
    # 1.2、合并文档
    merge_json_file(files=merge_input_files, output_file=merge_output_file)
    # 1.3、对每日新闻数据评分
    get_scores(news_data_file=news_data_file, device=device, saved_score_file=saved_score_file, save_to_csv=True)
    # 1.4、爬取股票数据
    fetch_price(api_key=API_KEY, ticker=ticker)
    # 1.5、处理股票数据
    fetch_stock(ticker=ticker)

def train():
    pre_train()
    # 要用wandb自动调参的话先去注册一个，拿一个api
    sweep_id = wandb.sweep(sweep_config, project=f"{ticker}_stock_forecast")  # 这里改你的名称，下面的链接点进去可以看训练情况
    # 使用wandb进行训练
    wandb.agent(sweep_id, training, count=10)  # 这里你可以看情况改，5次10次都可以

def test():
    """测试"""
    pred_df = predict_future(best_model_path, device=device)
    """画图"""
    visual()


if __name__ == '__main__':
    """0、初始化参数"""
    # 静态参数从settings中配置并获取
    # 注意用GPU,不知道为什么不改的话下面的代码都是默认用CPU，CPU还是有点慢,下面的代码有的地方我也稍微改了一下
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """爬数据+预处理数据"""
    fetch()

    """加载数据，模型训练"""
    train()

    """测试+可视化"""
    test()
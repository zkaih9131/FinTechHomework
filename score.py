import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import json

def load_finbert(device):
    tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone").to(device) #这里是在用huggingface加载FinBert，如果你是本地环境或者下得慢的话，可以先自己想办法这个模型下载下来
    return tokenizer, model

def get_sentiment(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs).logits
    scores = softmax(outputs, dim=1).cpu().numpy()[0]
    return scores[2] - scores[0]

#这里在用Finbert给新闻打分，它是每天所有的新闻放在一起算个平均分，花费的时间看你新闻的量
def process_sentiment(news_data, tokenizer, model, device):
    """使用 Finbert 给新闻打分，每天一个分数"""
    sentiment_scores = []
    for date, articles in news_data.items():
        print(date, end = "\r")
        scores = []
        for article in articles:
            # text = " ".join([article.get("title", ""), article.get("description", ""), article.get("content", "")])
            text = article.get("title", "")
            score = get_sentiment(text, tokenizer, model, device)
            scores.append(score)
        sentiment_scores.append({
            'date': date,
            'sentiment_score': np.mean(scores) if scores else 0.0
        }) #如果当天没有新闻就是0分
    return sentiment_scores

def get_scores(news_data_file: str, device, saved_score_file: str, save_to_csv: bool = True):
    with open(news_data_file, 'r', encoding='utf-8') as f:
        news_data = json.load(f)

    tokenizer, finbert_model = load_finbert(device=device)

    sentiment_scores = process_sentiment(news_data, tokenizer, finbert_model, device=device)

    if save_to_csv and saved_score_file is not None:
        sentiment_df = pd.DataFrame(sentiment_scores)
        sentiment_df.to_csv(saved_score_file, index=False)

        print(f"saved: {saved_score_file}")
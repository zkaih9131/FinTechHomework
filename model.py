import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#注意用GPU,不知道为什么不改的话下面的代码都是默认用CPU，CPU还是有点慢,下面的代码有的地方我也稍微改了一下

def load_finbert():
    tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone").to(device) #这里是在用huggingface加载FinBert，如果你是本地环境或者下得慢的话，可以先自己想办法这个模型下载下来
    return tokenizer, model

def get_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs).logits
    scores = softmax(outputs, dim=1).cpu().numpy()[0]
    return scores[2] - scores[0]

#这里在用Finbert给新闻打分，它是每天所有的新闻放在一起算个平均分，花费的时间看你新闻的量

def process_sentiment(news_data, tokenizer, model):
    sentiment_scores = []
    for date, articles in news_data.items():
        print(date, end = "\r")
        scores = []
        for article in articles:
            # text = " ".join([article.get("title", ""), article.get("description", ""), article.get("content", "")])
            text = article.get("title", "")
            score = get_sentiment(text, tokenizer, model)
            scores.append(score)
        sentiment_scores.append({
            'date': date,
            'sentiment_score': np.mean(scores) if scores else 0.0
        }) #如果当天没有新闻就是0分
    return sentiment_scores

with open('gnews_master.json', 'r') as f:
    news_data = json.load(f)

tokenizer, finbert_model = load_finbert()

sentiment_scores = process_sentiment(news_data, tokenizer, finbert_model)

sentiment_df = pd.DataFrame(sentiment_scores)
sentiment_df.to_csv('sentiment_scores_title.csv', index=False)

print("saved: sentiment_scores_title.csv")
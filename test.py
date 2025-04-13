import joblib
from pyecharts import options as opts
from pyecharts.charts import Line

from model import StockPriceLSTM
from score import *
from settings import *
from train import pre_train


def predict_future(model_path, device):
    merged_data, _, X_days, Y_days = pre_train()
    """测试结果"""
    model = StockPriceLSTM(1, 64, 2).to(device)  # 这里要改成你跑出来的那个模型的结构
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    scaler_X = joblib.load("scaler_X.save")
    scaler_y = joblib.load("scaler_y.save")

    full_data = merged_data.set_index("date")
    predict_dates = pd.date_range(start=pd.to_datetime("2025-03-01"), end=pd.to_datetime("2025-04-08"))# 预测期间就都是这一段吧，大家保持一致

    predictions = []
    for current_date in predict_dates:
        past_window = full_data.loc[:current_date - pd.Timedelta(days=1)].tail(max(X_days, Y_days))
        if len(past_window) < max(X_days, Y_days):
            continue

        closing_seq = past_window["closingValue"].values[-X_days:]
        sentiment_seq = past_window["sentiment_score"].values[-Y_days:]

        if len(closing_seq) < X_days or len(sentiment_seq) < Y_days:
            continue

        input_vector = np.concatenate([closing_seq, sentiment_seq]).reshape(1, -1)
        input_scaled = scaler_X.transform(input_vector).reshape(1, -1, 1)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)

        with torch.no_grad():
            predicted_scaled = model(input_tensor).cpu().item()
            predicted_real = scaler_y.inverse_transform([[predicted_scaled]])[0][0]
            predictions.append((current_date.strftime("%Y-%m-%d"), predicted_real))

    pred_df = pd.DataFrame(predictions, columns=["date", "predicted_closingValue"])
    pred_df.to_csv(f"future_predicted_{ticker}.csv", index=False)
    print(f"saved: future_predicted_{ticker}.csv")
    return pred_df

def visual():
    # 这里开始做数据可视化
    pred_df = pd.read_csv(f"future_predicted_{ticker}.csv", parse_dates=["date"])
    real_df = pd.read_csv(f"{ticker}_trend.csv", parse_dates=["date"])
    real_df = real_df[real_df["date"].isin(pred_df["date"])][["date", "closingValue"]]

    df = pd.merge(pred_df, real_df, on="date")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    line = (
        Line()
        .add_xaxis(df["date"].tolist())
        .add_yaxis("真实股价", df["closingValue"].round(2).tolist(), is_smooth=True)
        .add_yaxis("预测股价", df["predicted_closingValue"].round(2).tolist(), is_smooth=True,
                   linestyle_opts=opts.LineStyleOpts(type_="dashed"))
        .set_global_opts(
            title_opts=opts.TitleOpts(title=f"{ticker} 预测"),  # 只改这里的和上面读取文件里的股票名字，其他不要动
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            xaxis_opts=opts.AxisOpts(type_="category", axislabel_opts=opts.LabelOpts(rotate=45)),
            yaxis_opts=opts.AxisOpts(name="收盘价（USD）"),
            datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
            legend_opts=opts.LegendOpts(pos_top="5%")
        )
    )

    line.render(f"{ticker}_prediction_chart.html")
    print(f"已保存 {ticker}_prediction_chart.html")
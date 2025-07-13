import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
from io import StringIO
from pytrends.request import TrendReq
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# پیکربندی صفحه
st.set_page_config(page_title=" پیش‌بینی نرخ دلار آزاد تهران 📈", layout="wide")
st.markdown("""
---
📈 © 2025 Dr. Farhadi. All rights reserved.  
This application was developed by **Dr. Farhadi**, Ph.D. in *Economics (Econometrics)* and *Data Science*.  
All trademarks and intellectual property are protected. ™
""")
st.title("📈 پیش‌بینی نرخ دلار آزاد 📈")

# آدرس فایل ترندز در GitHub
GITHUB_TRENDS_CSV_URL = (
    'https://raw.githubusercontent.com/AZFARHAD24511/exchange_rates_IRAN/main/'
    'predict/google_trends_daily.csv'
)
KEYWORDS = ['خرید دلار', 'فروش دلار', 'دلار فردایی']

# بارگذاری داده‌های دلار آزاد از API
@st.cache_data(ttl=3600)
def load_usd_data():
    ts = int(datetime.now().timestamp() * 1000)
    url = (
        f"https://api.tgju.org/v1/market/indicator/"
        f"summary-table-data/price_dollar_rl?period=all&mode=full&ts={ts}"
    )
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    data = r.json().get('data', [])
    records = []
    for row in data:
        try:
            price = float(
                row[0].replace(',', '')
                      .replace('<span class="high" dir="ltr">', '')
                      .replace('</span>', '')
            )
            date = datetime.strptime(row[6], "%Y/%m/%d")
            records.append({'date': date, 'price': price})
        except:
            continue
    df = pd.DataFrame(records).set_index('date').sort_index()
    return df

# بارگذاری داده‌های Google Trends از GitHub
@st.cache_data(ttl=3600)
def load_trends_csv():
    r = requests.get(GITHUB_TRENDS_CSV_URL)
    df = pd.read_csv(StringIO(r.text), parse_dates=['date'])
    return df.set_index('date').sort_index()

# گرفتن داده‌های ناقص Google Trends
@st.cache_data(
    ttl=3600,
    hash_funcs={pd.DatetimeIndex: lambda idx: idx.astype(str).tolist()}
)
def fetch_missing_trends(missing_dates, geo='IR'):
    # اگر missing_dates از نوع tuple رشته‌ای باشد، تبدیل به DatetimeIndex می‌کنیم
    if not isinstance(missing_dates, pd.DatetimeIndex):
        missing_dates = pd.to_datetime(list(missing_dates))
    pytrends = TrendReq(hl='fa', tz=330)
    df_list = []
    start, end = missing_dates.min(), missing_dates.max()
    timeframe = f"{start.strftime('%Y-%m-%d')} {end.strftime('%Y-%m-%d')}"
    for kw in KEYWORDS:
        pytrends.build_payload([kw], timeframe=timeframe, geo=geo)
        tmp = pytrends.interest_over_time()
        if not tmp.empty:
            df_list.append(tmp[kw].rename(kw))
    if df_list:
        df_new = pd.concat(df_list, axis=1).loc[missing_dates]
        return df_new.apply(lambda x: x / x.max() * 100)
    return pd.DataFrame(index=missing_dates)

# بارگذاری و ترکیب داده‌ها
with st.spinner("در حال بارگذاری داده‌ها..."):
    usd_df = load_usd_data()
    trends_df = load_trends_csv()

# استفاده از دو سال اخیر
two_years_ago = datetime.now() - timedelta(days=730)
usd_df = usd_df[usd_df.index >= two_years_ago]
trends_df = trends_df[trends_df.index >= two_years_ago]

# پر کردن تاریخ‌های ناقص
missing = usd_df.index.difference(trends_df.index)
if not missing.empty:
    # برای اطمینان، missing را به tuple از رشته‌ها تبدیل می‌کنیم
    missing_tuple = tuple(date.strftime('%Y-%m-%d') for date in missing)
    new_trends = fetch_missing_trends(missing_tuple)
    trends_df = pd.concat([trends_df, new_trends]).sort_index()
    trends_df = trends_df.reindex(usd_df.index).ffill().bfill()

# ادغام داده‌ها
df = pd.merge(usd_df, trends_df, left_index=True, right_index=True, how='inner').ffill().bfill()
price_series = df['price']

# مدل ARIMA (1,0,1) با statsmodels
model = ARIMA(price_series, order=(1, 0, 1))
model_fit = model.fit()

# پیش‌بینی دو روز آینده
forecast_vals = model_fit.forecast(steps=2)
forecast_dates = [price_series.index[-1] + timedelta(days=i) for i in range(1, 3)]

# محاسبه دقت مدل
preds_in = model_fit.predict(start=0, end=len(price_series)-1)
mae = mean_absolute_error(price_series, preds_in)
mape = mean_absolute_percentage_error(price_series, preds_in) * 100

# استخراج p-value ضرایب
try:
    pvals = model_fit.pvalues
    pval_str = ', '.join([f'{name}: {val:.4f}' for name, val in pvals.items()])
except Exception:
    pval_str = "در دسترس نیست"

# نمایش دقت و p-values
st.info(f"MAE: {mae:,.2f}    MAPE: {mape:.2f}%    P-values: {pval_str}")
st.success(f"🔮 نرخ دلار برای {forecast_dates[0].date()}: {forecast_vals.iloc[0]:,.0f} ریال")
st.success(f"🔮 نرخ دلار برای {forecast_dates[1].date()}: {forecast_vals.iloc[1]:,.0f} ریال")
st.success(f"🔮 نرخ دلار برای {forecast_dates[2].date()}: {forecast_vals.iloc[2]:,.0f} ریال")

# نمایش نمودار
st.subheader("📊 Historical Data & 2-Day Forecast")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(price_series.index, price_series.values, label='Historical')
ax.axvline(price_series.index[-1], linestyle='--')
for i, (d, v) in enumerate(zip(forecast_dates, forecast_vals), start=1):
    ax.scatter(d, v)
    ax.annotate(
        f'Day+{i}: {v:,.0f}',
        xy=(d, v), xytext=(0, 10),
        textcoords='offset points', ha='center',
        arrowprops=dict(arrowstyle='->')
    )
ax.set_title('USD Free Market Rate Forecast')
ax.grid(True)
st.pyplot(fig)

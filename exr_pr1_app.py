import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

st.title("📈 پیش‌بینی سری زمانی قیمت با ARIMA (بدون pmdarima)")

# آپلود داده‌ها
uploaded_file = st.file_uploader("یک فایل CSV آپلود کنید", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # بررسی اولیه ستون‌ها
    st.subheader("پیش‌نمایش داده‌ها")
    st.write(df.head())

    # انتخاب ستون تاریخ و قیمت
    date_col = st.selectbox("ستون تاریخ را انتخاب کنید", df.columns)
    price_col = st.selectbox("ستون قیمت را انتخاب کنید", df.columns)

    # تبدیل به دیتای سری زمانی
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col)
    df.set_index(date_col, inplace=True)
    price_series = df[price_col].dropna()

    st.line_chart(price_series, height=250, use_container_width=True)

    # انتخاب پارامترهای ARIMA
    st.subheader("تنظیمات مدل ARIMA")
    p = st.number_input("p (Auto-Regressive)", min_value=0, max_value=5, value=1)
    d = st.number_input("d (Differencing)", min_value=0, max_value=2, value=1)
    q = st.number_input("q (Moving Average)", min_value=0, max_value=5, value=1)

    if st.button("اجرای مدل و پیش‌بینی"):
        try:
            model = ARIMA(price_series, order=(p, d, q))
            model_fit = model.fit()

            # پیش‌بینی آینده
            steps = st.slider("تعداد دوره‌های پیش‌بینی", min_value=1, max_value=30, value=7)
            forecast = model_fit.forecast(steps=steps)

            st.subheader("پیش‌بینی آینده")
            st.line_chart(forecast)

            # ارزیابی مدل
            preds_in = model_fit.predict(start=price_series.index[0], end=price_series.index[-1])
            mae = mean_absolute_error(price_series, preds_in)
            mape = mean_absolute_percentage_error(price_series, preds_in) * 100

            pvals = model_fit.pvalues
            pval_str = ', '.join(f"{name}: {val:.4f}" for name, val in pvals.items())

            st.info(f"MAE: {mae:,.2f}    |    MAPE: {mape:.2f}%\n\nP-values: {pval_str}")

            # نمودار واقعی vs پیش‌بینی‌شده
            fig, ax = plt.subplots()
            ax.plot(price_series, label="واقعی")
            ax.plot(preds_in, label="پیش‌بینی داخل نمونه")
            ax.legend()
            ax.set_title("پیش‌بینی درون نمونه ARIMA")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"خطا در اجرای مدل: {e}")

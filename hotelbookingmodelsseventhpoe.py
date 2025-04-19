import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Streamlit page configuration
st.set_page_config(page_title="Time Series Forecasting App", layout="wide")
st.title("📊 Time Series Forecasting Web App")

# --- Upload File ---
uploaded_file = st.file_uploader("Upload Time Series CSV", type=["csv"])

if uploaded_file:
    # Load dataset
    df = pd.read_csv(uploaded_file)

    # Fill missing values
    df['children'] = df['children'].fillna(0)
    df['country'] = df['country'].fillna('Unknown')
    df['agent'] = df['agent'].fillna(0)
    df['company'] = df['company'].fillna(0)

    # Combine year, month, day into a datetime
    df['arrival_date'] = pd.to_datetime(
        df['arrival_date_year'].astype(str) + '-' +
        df['arrival_date_month'].apply(lambda x: str(list(calendar.month_name).index(x))) + '-' +
        df['arrival_date_day_of_month'].astype(str)
    )

    # Set datetime index
    df.set_index('arrival_date', inplace=True)

    # Compute daily bookings
    df['total_bookings'] = (df['is_canceled'] == 0).astype(int)
    daily_bookings = df.resample('D').sum()['total_bookings']

    # Plot daily bookings
    st.subheader("Daily Hotel Bookings (Non-Canceled)")
    st.line_chart(daily_bookings)

    # ADF test for stationarity
    result = adfuller(daily_bookings.dropna())
    st.write("ADF Statistic:", result[0])
    st.write("p-value:", result[1])

    # --- Time Series Decomposition ---
    method = st.selectbox("Select Decomposition Method", ["Additive", "Multiplicative"])

    if st.button("Decompose"):
        if method == "Additive":
            add_decomp = seasonal_decompose(daily_bookings, model='additive', period=365)
            fig_add = add_decomp.plot()
            fig_add.suptitle('Additive Decomposition', fontsize=16, y=1.02)
            plt.tight_layout()
            st.pyplot(fig_add)
        else:
            mult_decomp = seasonal_decompose(daily_bookings.replace(0, 1), model='multiplicative', period=365)
            fig_mult = mult_decomp.plot()
            fig_mult.suptitle('Multiplicative Decomposition', fontsize=16, y=1.02)
            plt.tight_layout()
            st.pyplot(fig_mult)

    # --- Forecasting ---
    st.subheader("Forecasting Models")
    model_choice = st.selectbox("Select Forecasting Model", ["ARIMA", "Exponential Smoothing", "Prophet", "LSTM"])
    forecast_period = st.slider("Forecast Period (Days)", 1, 90, 30)

    if st.button("Forecast"):
        train = daily_bookings.iloc[:-forecast_period]
        test = daily_bookings.iloc[-forecast_period:]

        if model_choice == "ARIMA":
            arima_model = ARIMA(train, order=(5, 1, 2))
            arima_result = arima_model.fit()
            arima_forecast = arima_result.forecast(steps=forecast_period)
            st.line_chart(pd.DataFrame(arima_forecast, index=test.index))

        elif model_choice == "Exponential Smoothing":
            ets_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=365).fit()
            ets_forecast = ets_model.forecast(forecast_period)
            st.line_chart(pd.DataFrame(ets_forecast, index=test.index))

        elif model_choice == "Prophet":
            df_prophet = daily_bookings.reset_index().rename(columns={'arrival_date': 'ds', 'total_bookings': 'y'})
            prophet_model = Prophet()
            prophet_model.fit(df_prophet.iloc[:-forecast_period])
            future = prophet_model.make_future_dataframe(periods=forecast_period)
            forecast = prophet_model.predict(future)
            st.line_chart(forecast.set_index('ds')['yhat'])

        elif model_choice == "LSTM":
            scaler = MinMaxScaler()
            scaled_series = scaler.fit_transform(daily_bookings.values.reshape(-1, 1))

            def create_sequences(series, window=30):
                X, y = [], []
                for i in range(window, len(series)):
                    X.append(series[i-window:i])
                    y.append(series[i])
                return np.array(X), np.array(y)

            X, y = create_sequences(scaled_series)
            X_train, X_test = X[:-forecast_period], X[-forecast_period:]
            y_train, y_test = y[:-forecast_period], y[-forecast_period:]

            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            lstm_model = Sequential([
                Input(shape=(X_train.shape[1], 1)),
                LSTM(50, activation='relu'),
                Dense(1)
            ])
            lstm_model.compile(optimizer='adam', loss='mse')
            lstm_model.fit(X_train, y_train, epochs=10, verbose=1)

            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            lstm_forecast = lstm_model.predict(X_test)
            lstm_forecast = scaler.inverse_transform(lstm_forecast)
            st.line_chart(pd.DataFrame(lstm_forecast.flatten(), index=test.index))

        # --- Evaluation ---
        st.subheader("Evaluation Metrics")
        def evaluate(y_true, y_pred):
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            return rmse, mae, mape

        if model_choice == "ARIMA":
            rmse, mae, mape = evaluate(test, arima_forecast)
        elif model_choice == "Exponential Smoothing":
            rmse, mae, mape = evaluate(test, ets_forecast)
        elif model_choice == "Prophet":
            rmse, mae, mape = evaluate(test, forecast['yhat'].iloc[-forecast_period:].values)
        elif model_choice == "LSTM":
            rmse, mae, mape = evaluate(test.values, lstm_forecast.flatten())

        st.write("RMSE:", rmse)
        st.write("MAE:", mae)
        st.write("MAPE:", mape)
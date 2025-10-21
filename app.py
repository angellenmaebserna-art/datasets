# app.py — FINAL CLEAN VERSION (Forecasts moved to 🔮 Predictions tab)
import os
import itertools
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.environ.setdefault('PROPHET_BACKEND', 'CMDSTANPY')
plt.ioff()

try:
    import cmdstanpy
except Exception:
    cmdstanpy = None

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# -------------------- 🌊 THEME --------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom, #f6f9fb 0%, #e5eef3 50%, #437290 100%);
    overflow: hidden;
}
section[data-testid="stSidebar"] {
    background-color: #DDE3EC !important;
}
h1, h2, h3 {
    color: #01579b !important;
    text-shadow: 0 2px 4px rgba(255,255,255,0.6);
}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD CSV --------------------
st.sidebar.title("Select Dataset")

csv_files = ["merged_microplastic_data.csv"]
datasets = {}
for file in csv_files:
    try:
        df_temp = pd.read_csv(file)
        df_temp.columns = df_temp.columns.str.strip().str.replace(" ", "_").str.title()
        datasets[file.split("/")[-1].replace(".csv", "")] = df_temp
    except FileNotFoundError:
        st.warning(f"⚠️ File not found: {file}")

if len(datasets) == 0:
    st.error("No datasets found.")
    st.stop()

selected_dataset = st.sidebar.selectbox("Select Dataset / Place", list(datasets.keys()))
df = datasets[selected_dataset]

lat_col = next((c for c in df.columns if "lat" in c.lower()), None)
lon_col = next((c for c in df.columns if "lon" in c.lower() or "long" in c.lower()), None)

# -------------------- NAVIGATION --------------------
menu = st.sidebar.radio(
    "Go to:",
    ["🏠 Dashboard", "🌍 Heatmap", "📊 Analytics", "🔮 Predictions", "📜 Reports"]
)

# -------------------- DASHBOARD --------------------
if menu == "🏠 Dashboard":
    st.title(f"🏠 AI-Driven Microplastic Dashboard — {selected_dataset}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", len(df))
    c2.metric("Columns", len(df.columns))
    c3.metric("Source", "Local CSV")
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

# -------------------- HEATMAP --------------------
elif menu == "🌍 Heatmap":
    st.title(f"🌍 Microplastic Heatmap — {selected_dataset}")
    if lat_col and lon_col:
        map_df = df.rename(columns={lat_col: "latitude", lon_col: "longitude"})
        map_df["latitude"] = pd.to_numeric(map_df["latitude"], errors="coerce")
        map_df["longitude"] = pd.to_numeric(map_df["longitude"], errors="coerce")
        if not map_df[["latitude", "longitude"]].dropna().empty:
            st.map(map_df[["latitude", "longitude"]].dropna())
        else:
            st.warning("No valid coordinates found.")
    else:
        st.error("Latitude/Longitude columns not detected.")

# -------------------- ANALYTICS --------------------
elif menu == "📊 Analytics":
    st.title(f"📊 Analytics Overview — {selected_dataset}")
    st.write("Descriptive and correlation overview of the dataset.")

    # --- Basic descriptive ---
    st.subheader("📋 Descriptive Statistics")
    st.dataframe(df.describe())

    if "Microplastic_Level" in df.columns and "Year" in df.columns:
        st.subheader("📈 Mean Microplastic Level Over Years")
        yearly = df.groupby("Year")["Microplastic_Level"].mean().reset_index()
        fig, ax = plt.subplots()
        sns.lineplot(x="Year", y="Microplastic_Level", data=yearly, marker="o", ax=ax)
        st.pyplot(fig)

        st.subheader("📊 Correlation Matrix")
        corr = df.select_dtypes(include=[np.number]).corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("No 'Year' or 'Microplastic_Level' columns found.")

# -------------------- PREDICTIONS --------------------
elif menu == "🔮 Predictions":
    st.title(f"🔮 Forecasting & Prediction — {selected_dataset}")

    model_choice = st.selectbox("Select Model:", ["Prophet", "SARIMA", "ARIMA", "SES"])
    target_col = st.selectbox("Select target column:", [c for c in df.columns if "microplastic" in c.lower()])

    # --- PREPARE YEARLY DATA ---
    df = df.dropna(subset=[target_col])
    year_col = [c for c in df.columns if c.lower() == "year"][0]
    yearly = df.groupby(year_col)[target_col].mean().reset_index()

    # ---------------- PROPHET ----------------
    if model_choice == "Prophet":
        from prophet import Prophet
        from sklearn.metrics import mean_absolute_error
        prophet_df = yearly.rename(columns={year_col: "ds", target_col: "y"})
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"].astype(int).astype(str) + "-01-01")
        m = Prophet(yearly_seasonality=True)
        m.fit(prophet_df)
        future = m.make_future_dataframe(periods=5, freq="Y")
        forecast = m.predict(future)
        st.subheader("📊 Prophet Forecast")
        st.pyplot(m.plot(forecast))
        st.pyplot(m.plot_components(forecast))

    # ---------------- SARIMA ----------------
    elif model_choice == "SARIMA":
        import statsmodels.api as sm
        ts = yearly.set_index(year_col)[target_col]
        p = d = q = [0, 1]
        pdq = list(itertools.product(p, d, q))
        best_aic, best_res, best_order = np.inf, None, None
        for order in pdq:
            try:
                model = sm.tsa.statespace.SARIMAX(ts, order=order, enforce_stationarity=False)
                results = model.fit(disp=False)
                if results.aic < best_aic:
                    best_aic, best_res, best_order = results.aic, results, order
            except:
                continue
        if best_res is not None:
            st.success(f"Best SARIMA order: {best_order} (AIC={best_aic:.2f})")
            forecast = best_res.get_forecast(steps=5)
            forecast_mean = forecast.predicted_mean
            forecast_ci = forecast.conf_int()
            fig, ax = plt.subplots()
            ts.plot(ax=ax, label="Observed")
            forecast_mean.plot(ax=ax, color="r", label="Forecast")
            ax.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color="pink", alpha=0.3)
            ax.legend()
            st.pyplot(fig)

    # ---------------- ARIMA ----------------
    elif model_choice == "ARIMA":
        from statsmodels.tsa.arima.model import ARIMA
        ts = yearly[target_col]
        model = ARIMA(ts, order=(5, 1, 0))
        fit = model.fit()
        forecast = fit.forecast(steps=5)
        st.subheader("🔴 ARIMA Forecast")
        st.line_chart(forecast)

    # ---------------- SES ----------------
    elif model_choice == "SES":
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing
        ts = yearly[target_col]
        model = SimpleExpSmoothing(ts)
        fit = model.fit()
        forecast = fit.forecast(steps=5)
        st.subheader("🟢 Simple Exponential Smoothing Forecast")
        st.line_chart(forecast)

# -------------------- REPORTS --------------------
elif menu == "📜 Reports":
    st.title("📜 Reports")
    st.dataframe(df.describe())
    if "future_df" in st.session_state:
        st.dataframe(st.session_state["future_df"])
    else:
        st.info("Run a forecast first to generate report data.")

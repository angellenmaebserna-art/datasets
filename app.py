import os
import itertools
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Try to set Prophet backend environment to CMDSTANPY to reduce backend errors
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

# -------------------- 🌊 3D Pastel Water Theme --------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom, #f6f9fb 0%, #e5eef3 50%, #437290 100%);
}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD MULTIPLE CSV FILES --------------------
st.sidebar.title("Select Dataset")
csv_files = ["merged_microplastic_data.csv"]
datasets = {}
for file in csv_files:
    try:
        df_temp = pd.read_csv(file)
        df_temp.columns = df_temp.columns.str.strip().str.replace(" ", "_").str.title()
        datasets[file.split("/")[-1].replace(".csv","")] = df_temp
    except FileNotFoundError:
        st.warning(f"⚠️ File not found: {file}")

if not datasets:
    st.error("No datasets found.")
    st.stop()

selected_dataset = st.sidebar.selectbox("Select Dataset", list(datasets.keys()))
df = datasets[selected_dataset]

# Detect coordinates
lat_col, lon_col = None, None
for col in df.columns:
    if "lat" in col.lower(): lat_col = col
    if "lon" in col.lower() or "long" in col.lower(): lon_col = col

# -------------------- NAVIGATION --------------------
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to:", ["🏠 Dashboard", "🌍 Heatmap", "📊 Analytics", "🔮 Predictions", "📜 Reports"])

# -------------------- DASHBOARD --------------------
if menu == "🏠 Dashboard":
    st.title(f"🏠 Dashboard — {selected_dataset}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Available Columns", len(df.columns))
    col3.metric("Data Source", "Local CSV")
    st.dataframe(df.head())

# -------------------- HEATMAP --------------------
elif menu == "🌍 Heatmap":
    st.title("🌍 Microplastic Heatmap")
    if lat_col and lon_col:
        map_df = df.rename(columns={lat_col: "latitude", lon_col: "longitude"})
        map_df["latitude"] = pd.to_numeric(map_df["latitude"], errors="coerce")
        map_df["longitude"] = pd.to_numeric(map_df["longitude"], errors="coerce")
        st.map(map_df[["latitude", "longitude"]].dropna())
    else:
        st.error("No latitude/longitude columns found.")

# -------------------- ANALYTICS --------------------
elif menu == "📊 Analytics":
    st.title(f"📊 Analytics of {selected_dataset}")
    st.write("Descriptive, correlation, and classification overview.")

    # Rename columns
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ["year", "yr"]: col_map[c] = "Year"
        elif "microplastic" in cl: col_map[c] = "Microplastic_Level"
        elif "lat" in cl: col_map[c] = "Latitude"
        elif "long" in cl: col_map[c] = "Longitude"
        elif "place" in cl: col_map[c] = "Place"
        elif "site" in cl: col_map[c] = "Site"
    if col_map: df = df.rename(columns=col_map)

    if not {"Year", "Microplastic_Level"}.issubset(df.columns):
        st.warning("Some expected columns missing.")
    else:
        st.subheader("📅 Yearly Aggregation")
        yearly_microplastic = df.groupby("Year")["Microplastic_Level"].mean().reset_index()
        st.dataframe(yearly_microplastic.head())

        # --- EDA Visualizations ---
        try:
            st.subheader("📈 Mean Microplastic Level Over Years")
            fig, ax = plt.subplots(figsize=(8,4))
            sns.lineplot(x="Year", y="Microplastic_Level", data=yearly_microplastic, marker="o", ax=ax)
            st.pyplot(fig)
        except: pass

        st.subheader("🔎 Distribution & Boxplot")
        fig, ax = plt.subplots(1,2, figsize=(10,4))
        sns.histplot(df["Microplastic_Level"], kde=True, ax=ax[0])
        sns.boxplot(x=df["Microplastic_Level"], ax=ax[1])
        st.pyplot(fig)

        st.subheader("📊 Correlation Matrix")
        num_cols = [c for c in ["Latitude","Longitude","Year","Microplastic_Level"] if c in df.columns]
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        # --- Classification Example (Preserved) ---
        st.markdown("### 🧠 Preprocessing & Classification (Ma'am's code preserved)")
        try:
            from sklearn.preprocessing import OneHotEncoder
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import classification_report, accuracy_score

            features = [c for c in ["Site","Place","Latitude","Longitude","Year"] if c in df.columns]
            if len(features) >= 2:
                X = df[features].copy()
                y = df["Microplastic_Level"].copy()
                cat_feats = [c for c in ["Site","Place"] if c in X.columns]
                if cat_feats:
                    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                    X = X.drop(cat_feats, axis=1).join(pd.DataFrame(enc.fit_transform(df[cat_feats]), index=X.index))
                low, med = y.quantile(0.5), y.quantile(0.9)
                y_cat = y.apply(lambda v: "Low" if v<=low else "Medium" if v<=med else "High")
                X_train,X_test,y_train,y_test=train_test_split(X,y_cat,test_size=0.2,random_state=42)
                rf=RandomForestClassifier(n_estimators=100,random_state=42)
                rf.fit(X_train,y_train)
                y_pred=rf.predict(X_test)
                st.text("Classification Report:")
                st.text(classification_report(y_test,y_pred))
                st.write(f"Accuracy: {accuracy_score(y_test,y_pred):.4f}")
        except Exception as e:
            st.warning(f"Classification block failed: {e}")

# -------------------- PREDICTIONS --------------------
elif menu == "🔮 Predictions":
    st.title(f"🔮 Prediction & Forecasting — {selected_dataset}")
    st.markdown("<br>", unsafe_allow_html=True)

    model_choice = st.selectbox("Select forecasting model:", ["Random Forest", "ARIMA & SES", "Prophet", "SARIMA"])

    target_col = "Microplastic_Level" if "Microplastic_Level" in df.columns else st.selectbox("Select Target", df.columns)
    df_model = df.dropna(subset=[target_col])

    # ---------------- RANDOM FOREST ----------------
    if model_choice == "Random Forest":
        st.markdown("### 🌳 Random Forest Forecasting (Regression)")
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error

        features = df_model.select_dtypes(include=[np.number]).drop(columns=[target_col], errors="ignore")
        X_train,X_test,y_train,y_test=train_test_split(features,df_model[target_col],test_size=0.2,random_state=42)
        rf=RandomForestRegressor(n_estimators=300,max_depth=10,random_state=42)
        rf.fit(X_train,y_train)
        y_pred=rf.predict(X_test)
        r2=r2_score(y_test,y_pred); rmse=np.sqrt(mean_squared_error(y_test,y_pred)); mae=mean_absolute_error(y_test,y_pred)
        st.write(f"R²={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")
        fig,ax=plt.subplots(); ax.scatter(y_test,y_pred); ax.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'r--'); st.pyplot(fig)

    # ---------------- ARIMA & SES ----------------
    elif model_choice == "ARIMA & SES":
        st.markdown("### ⏳ Time Series Forecasting (ARIMA, SES) — Ma'am's code preserved")
        try:
            yearly=df.groupby("Year")[target_col].mean().reset_index()
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.holtwinters import SimpleExpSmoothing
            # ARIMA
            st.subheader("🔴 ARIMA Model")
            model=ARIMA(yearly[target_col],order=(5,1,0))
            res=model.fit()
            forecast=res.forecast(steps=3)
            st.dataframe(pd.DataFrame({'ARIMA Forecast':forecast}))
            # SES
            st.subheader("🟢 Simple Exponential Smoothing")
            ses=SimpleExpSmoothing(yearly[target_col]).fit()
            f_ses=ses.forecast(3)
            st.dataframe(pd.DataFrame({'SES Forecast':f_ses}))
            # Combined plot
            fig,ax=plt.subplots(figsize=(8,5))
            ax.plot(yearly["Year"],yearly[target_col],label="Historical")
            ax.plot(range(yearly["Year"].max()+1,yearly["Year"].max()+4),forecast,label="ARIMA")
            ax.plot(range(yearly["Year"].max()+1,yearly["Year"].max()+4),f_ses,label="SES")
            ax.legend(); st.pyplot(fig)
        except Exception as e:
            st.error(f"Forecast failed: {e}")

    # ---------------- PROPHET ----------------
    elif model_choice == "Prophet":
        st.markdown("### 🟣 Prophet Forecasting — Ma'am's code preserved")
        try:
            from prophet import Prophet
            yearly=df.groupby("Year")[target_col].mean().reset_index()
            yearly["ds"]=pd.to_datetime(yearly["Year"].astype(str)+"-01-01")
            yearly["y"]=yearly[target_col]
            m=Prophet(yearly_seasonality=True)
            m.fit(yearly[["ds","y"]])
            future=m.make_future_dataframe(periods=3,freq="Y")
            forecast=m.predict(future)
            st.dataframe(forecast[["ds","yhat","yhat_lower","yhat_upper"]].tail())
            st.pyplot(m.plot(forecast))
            st.pyplot(m.plot_components(forecast))
        except Exception as e:
            st.error(f"Prophet forecasting failed: {e}")

    # ---------------- SARIMA ----------------
    elif model_choice == "SARIMA":
        st.markdown("### 🔁 SARIMA — Ma'am's code preserved")
        try:
            import statsmodels.api as sm
            ts=df_model.set_index("Year")[target_col].astype(float)
            p=d=q=[0,1]; pdq=list(itertools.product(p,d,q))
            best_aic=np.inf; best=None; order_best=None
            for order in pdq:
                try:
                    res=sm.tsa.statespace.SARIMAX(ts,order=order).fit(disp=False)
                    if res.aic<best_aic: best_aic,best,order_best=res.aic,res,order
                except: pass
            if best is not None:
                st.success(f"Best order: {order_best}, AIC={best_aic:.2f}")
                pred=best.get_forecast(steps=5)
                ci=pred.conf_int()
                fig,ax=plt.subplots(figsize=(8,4))
                ts.plot(ax=ax,label="Observed")
                years=np.arange(ts.index.max()+1,ts.index.max()+6)
                ax.plot(years,pred.predicted_mean.values,label="Forecast",color="red")
                ax.fill_between(years,ci.iloc[:,0],ci.iloc[:,1],color="pink",alpha=0.3)
                ax.legend(); st.pyplot(fig)
        except Exception as e:
            st.error(f"SARIMA failed: {e}")

# -------------------- REPORTS --------------------
elif menu == "📜 Reports":
    st.title(f"📜 Reports Section of {selected_dataset}")
    st.write("Generate downloadable reports of analytics and predictions.")
    st.subheader("1️⃣ Summary Report")
    st.dataframe(df.describe())

    if "future_df" in st.session_state:
        future_df = st.session_state["future_df"]
        st.subheader("2️⃣ Forecast Results (2026–2030)")
        st.dataframe(future_df.style.format({"Predicted_Microplastic_Level": "{:.2f}"}))
        csv = future_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Forecast (CSV)",
            data=csv,
            file_name=f"{selected_dataset}_forecast_2026_2030.csv",
            mime="text/csv"
        )
    else:
        st.info("⚠️ No forecast data available yet. Please run Predictions tab first.")

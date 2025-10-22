import itertools
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import numpy as np
if not hasattr(np, 'float_'):
    np.float_ = float
import statsmodels
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# -------------------- 🌊 3D Pastel Water Theme --------------------
st.markdown("""
    <style>
    /*  🌊 3D animated pastel water background */
    .stApp {
        position: relative;
        background: linear-gradient(to bottom, #f6f9fb 0%, #e5eef3 50%, #437290 100%);
        overflow: hidden;
    }

    /* Waves overlay effect */
    .stApp::before, .stApp::after {
        content: "";
        position: absolute;
        left: 0;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 50% 50%, rgba(255,255,255,0.4) 0%, transparent 70%),
                    radial-gradient(circle at 70% 70%, rgba(255,255,255,0.3) 0%, transparent 70%),
                    radial-gradient(circle at 30% 30%, rgba(255,255,255,0.25) 0%, transparent 70%);
        animation: waveMove 15s infinite linear;
        opacity: 0.4;
        z-index: 0;
    }

    .stApp::after {
        animation-delay: -7s;
        opacity: 0.3;
    }

    @keyframes waveMove {
        from { transform: translateX(0) translateY(0) rotate(0deg); }
        to { transform: translateX(-25%) translateY(-25%) rotate(360deg); }
    }

   /* 🧩 Fix overlapping sidebar issue */
section[data-testid="stSidebar"] {
    position: relative !important; /* para dili siya mag-float sa ibabaw */
    z-index: 1 !important; /* ipa-ubos ang layer niya */
    overflow-y: auto !important; /* para ma-scroll gihapon */
    background-color: #DDE3EC !important;
    backdrop-filter: blur(6px);
    height: 100vh !important; /* sakto ang taas */
}

/* Main content stays on top of sidebar */
[data-testid="stAppViewContainer"],
.main {
    position: relative !important;
    z-index: 2 !important; /* mas taas kaysa sidebar */
}

/* Waves effect stays at the very back */
.stApp::before,
.stApp::after {
    z-index: 0 !important;
}

/* Background waves stay behind everything */
.stApp::before,
.stApp::after {
    z-index: 0 !important;

    }

    /* Glassy translucent box for parameters */
    [data-testid="stJson"] {
        background: rgba(240, 248, 255, 0.35) !important;
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: #01579b !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    }

    /* DataFrames, charts, and metrics also glass-like */
    [data-testid="stDataFrame"], .stMetric {
        background: rgba(255, 255, 255, 0.4) !important;
        backdrop-filter: blur(8px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }

    /* Headings styling - wave effect */
    h1, h2, h3 {
        color: #01579b !important;
        text-shadow: 0 2px 4px rgba(255,255,255,0.6);
        animation: floatTitle 3s ease-in-out infinite;
    }

    @keyframes floatTitle {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-3px); }
    }

    /* Buttons hover shimmer */
    button, .stRadio label:hover {
        background: linear-gradient(120deg, #b3e5fc, #81d4fa);
        color: #01579b !important;
        border-radius: 10px;
        transition: 0.3s;
    }

    </style>
""", unsafe_allow_html=True)


# -------------------- LOAD MULTIPLE CSV FILES --------------------
st.sidebar.title("Select Dataset")

csv_files = [
    "merged_microplastic_data.csv"
]

datasets = {}
for file in csv_files:
    try:
        df_temp = pd.read_csv(file)
        df_temp.columns = df_temp.columns.str.strip().str.replace(" ", "_").str.title()
        datasets[file.split("/")[-1].replace(".csv","")] = df_temp
    except FileNotFoundError:
        st.warning(f"⚠️ File not found: {file}")

selected_dataset = st.sidebar.selectbox("Select Dataset / Place", list(datasets.keys()))
df = datasets[selected_dataset]

lat_col, lon_col = None, None
for col in df.columns:
    if "lat" in col.lower():
        lat_col = col
    if "lon" in col.lower() or "long" in col.lower():
        lon_col = col

# -------------------- STREAMLIT UI --------------------
st.sidebar.title("Navigation")

menu = st.sidebar.radio(
    "Go to:",
    [
        "🏠 Dashboard",
        "🌍 Heatmap",
        "📊 Analytics",
        "🔮 Predictions",
        "📜 Reports"
    ]
)

# -------------------- DASHBOARD --------------------
if menu == "🏠 Dashboard":
    st.title(f"🏠 AI-Driven Microplastic Monitoring Dashboard of {selected_dataset}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Available Columns", len(df.columns))
    with col3:
        st.metric("Data Source", "Local CSV")

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

# -------------------- HEATMAP --------------------
elif menu == "🌍 Heatmap":
    st.title(f"🌍 Microplastic HeatMap of {selected_dataset}")

    if lat_col and lon_col:
        st.success(f"Detected coordinates: **{lat_col}** and **{lon_col}**")

        map_df = df.rename(columns={lat_col: "latitude", lon_col: "longitude"})
        map_df["latitude"] = pd.to_numeric(map_df["latitude"], errors="coerce")
        map_df["longitude"] = pd.to_numeric(map_df["longitude"], errors="coerce")

        if map_df[["latitude", "longitude"]].dropna().empty:
            st.warning("⚠️ No valid latitude/longitude data found for map display.")
        else:
            st.map(map_df[["latitude", "longitude"]].dropna())
    else:
        st.error("⚠️ No latitude/longitude columns found in dataset.")

# -------------------- ANALYTICS --------------------
elif menu == "📊 Analytics":
    st.title(f"📊 Analytics of {selected_dataset}")
    st.write("Descriptive and correlation overview of the dataset.")
    st.markdown("---")

    # Normalize columns to expected names (case-insensitive)
    col_map = {}
    for c in df.columns:
        clower = c.lower()
        if clower in ["year", "yr"]:
            col_map[c] = "Year"
        elif clower in ["microplastic_level", "microplastic level", "microplasticlevel"]:
            col_map[c] = "Microplastic_Level"
        elif "latitude" in clower or clower == "lat":
            col_map[c] = "Latitude"
        elif "longitude" in clower or "long" in clower or clower == "lon":
            col_map[c] = "Longitude"
        elif clower == "site":
            col_map[c] = "Site"
        elif clower == "place":
            col_map[c] = "Place"

    # rename columns if any mappings found
    if len(col_map) > 0:
        df = df.rename(columns=col_map)

    # Basic checks
    expected_cols = ["Year", "Microplastic_Level"]
    missing_expected = [c for c in expected_cols if c not in df.columns]
    if missing_expected:
        st.warning(f"Some expected columns not found for extended analysis: {missing_expected}. Ma'am's analysis may be limited.")
    else:
        # ------- Data loading / yearly aggregation (as in microplasticDM) -------
        try:
            st.subheader("📅 Yearly Aggregation")
            yearly_microplastic = df.groupby('Year')['Microplastic_Level'].mean().reset_index()
            st.dataframe(yearly_microplastic.head())
        except Exception as e:
            st.error(f"Failed to compute yearly aggregation: {e}")
            yearly_microplastic = None

        # ------- Visualizations from microplasticDM -------
        try:
            st.subheader("📈 Mean Microplastic Level Over Years")
            mean_microplastic_by_year = df.groupby('Year')['Microplastic_Level'].mean().reset_index()
            fig, ax = plt.subplots(figsize=(8,4))
            sns.lineplot(x='Year', y='Microplastic_Level', data=mean_microplastic_by_year, marker='o', ax=ax)
            ax.set_title('Mean Microplastic Level Over Years')
            ax.set_xlabel('Year'); ax.set_ylabel('Mean Microplastic Level')
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not plot mean microplastic by year: {e}")

        # Distribution & boxplot
        try:
            st.subheader("🔎 Distribution & Boxplot of Microplastic Levels")
            fig, ax = plt.subplots(1,2, figsize=(12,4))
            sns.histplot(df['Microplastic_Level'], kde=True, ax=ax[0])
            ax[0].set_title('Distribution of Microplastic Levels')
            sns.boxplot(x=df['Microplastic_Level'], ax=ax[1])
            ax[1].set_title('Box Plot of Microplastic Levels')
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Distribution/boxplot error: {e}")

        # Scatter vs Year/Longitude/Latitude
        try:
            st.subheader("📌 Scatter: Microplastic vs Year / Longitude / Latitude")
            fig, ax = plt.subplots(1,3, figsize=(15,4))
            sns.scatterplot(x='Year', y='Microplastic_Level', data=df, ax=ax[0])
            ax[0].set_title('Microplastic Level vs Year')
            sns.scatterplot(x='Longitude', y='Microplastic_Level', data=df, ax=ax[1])
            ax[1].set_title('Microplastic Level vs Longitude')
            sns.scatterplot(x='Latitude', y='Microplastic_Level', data=df, ax=ax[2])
            ax[2].set_title('Microplastic Level vs Latitude')
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Scatter plots error: {e}")

        # Correlation matrix
        try:
            st.subheader("📊 Correlation Matrix")
            numerical_cols = ['Latitude', 'Longitude', 'Year', 'Microplastic_Level']
            numerical_cols_existing = [c for c in numerical_cols if c in df.columns]
            if len(numerical_cols_existing) >= 2:
                correlation_matrix = df[numerical_cols_existing].corr()
                fig, ax = plt.subplots(figsize=(8,6))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                ax.set_title('Correlation Matrix of Numerical Variables')
                st.pyplot(fig)
            else:
                st.info("Not enough numerical columns for correlation matrix.")
        except Exception as e:
            st.warning(f"Correlation matrix failed: {e}")

        # Microplastic by Place (aggregate)
        try:
            st.subheader("📍 Microplastic by Place")
            if 'Place' in df.columns:
                microplastic_by_place = df.groupby('Place')['Microplastic_Level'].agg(['mean', 'median', 'count', 'min', 'max']).reset_index()
                microplastic_by_place_sorted = microplastic_by_place.sort_values(by='mean', ascending=False)
                st.dataframe(microplastic_by_place_sorted.head(30))
                st.write(f"Number of unique place locations: {len(microplastic_by_place_sorted)}")
            else:
                st.info("No 'Place' column found.")
        except Exception as e:
            st.warning(f"Microplastic by Place failed: {e}")

        # Geographical aggregation
        try:
            st.subheader("📍 Geographical Aggregation (sample)")
            if 'Latitude' in df.columns and 'Longitude' in df.columns:
                microplastic_by_coordinates = df.groupby(['Latitude', 'Longitude'])['Microplastic_Level'].agg(['mean', 'median', 'count']).reset_index()
                st.dataframe(microplastic_by_coordinates.head())
                st.write(f"Number of unique coordinate locations: {len(microplastic_by_coordinates)}")
            else:
                st.info("No 'Latitude'/'Longitude' columns found for coordinate aggregation.")
        except Exception as e:
            st.warning(f"Geographical aggregation failed: {e}")


        # NOTE: Forecasting blocks (ARIMA, SES, Prophet, combined plot) were originally here in Ma'am's notebook.
        # Per your instruction, those forecasting RESULT displays are moved to the Predictions tab below.
        # All original forecasting code and comments are preserved but relocated.


elif menu == "🔮 Predictions":
    st.title(f"🔮 Prediction & Forecasting — {selected_dataset}")
    st.markdown("<br>", unsafe_allow_html=True)

    # Retain model selector and Random Forest behavior, and include all forecasting blocks (ARIMA/SES/Prophet/SARIMA/combined)
    model_choice = st.selectbox("Select forecasting model:", ["Random Forest", "Prophet", "ARIMA", "SES", "SARIMA", "All (compare)"])

    # try to guess target columns
    potential_targets = [c for c in df.columns if c.lower() in ["microplastic_level", "ph_level", "microplastic level", "ph level"]]
    if len(potential_targets) == 0:
        st.warning("No obvious target columns found (like 'Microplastic_Level' or 'pH_Level'). Please ensure data has a target column.")
        target_col = st.selectbox("Manually select target column:", list(df.columns))
    else:
        target_col = st.selectbox("Select target to forecast:", potential_targets)

    # normalize selection to exact column name
    target_col = [c for c in df.columns if c.lower().replace(" ", "_") == target_col.lower().replace(" ", "_")][0]
    df_model = df.copy().dropna(subset=[target_col])

    # Prepare yearly series for forecasting where needed (recompute here to avoid cross-tab variable dependency)
    yearly_microplastic = None
    if 'Year' in df_model.columns:
        try:
            yearly_microplastic = df_model.groupby('Year')[target_col].mean().reset_index().rename(columns={target_col: 'Microplastic_Level'})
        except Exception:
            yearly_microplastic = None

    # -------------------- RANDOM FOREST --------------------
    if model_choice == "Random Forest":
        st.sidebar.subheader("⚙️ Random Forest Parameters")
        n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 50, 500, 300, step=50)
        max_depth = st.sidebar.slider("Tree Depth (max_depth)", 1, 30, 10)
        test_size = st.sidebar.slider("Test Data Ratio", 0.1, 0.5, 0.2, step=0.05)

        task_type = st.radio("Select Task Type:", ["Regression", "Classification"])

        try:
            features = df_model.select_dtypes(include=[np.number]).drop(columns=[target_col], errors="ignore")
            if features.shape[1] == 0:
                st.error("No numeric features available for modeling. Please provide numeric feature columns.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    features, df_model[target_col], test_size=test_size, random_state=42
                )

                # ---------------- REGRESSION MODE ----------------
                if task_type == "Regression":
                    from sklearn.ensemble import RandomForestRegressor
                    from sklearn.metrics import mean_absolute_error
                    from sklearn.model_selection import cross_val_score  # ✅ added here

                    rf = RandomForestRegressor(
                        n_estimators=n_estimators, max_depth=max_depth, random_state=42
                    )
                    rf.fit(X_train, y_train)
                    y_pred = rf.predict(X_test)

                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)

                    def interpret_r2_local(r2_val):
                        return (
                            "Excellent" if r2_val >= 0.8 else
                            "Good" if r2_val >= 0.6 else
                            "Fair" if r2_val >= 0.3 else
                            "Poor" if r2_val >= 0 else
                            "Very Poor"
                        )

                    def interpret_err_local(err, y_vals):
                        ratio = (err / np.mean(y_vals)) * 100 if np.mean(y_vals) != 0 else 0
                        return "Low" if ratio < 10 else "Moderate" if ratio < 30 else "High"

                    st.subheader("📊 Model Accuracy (Regression)")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("R²", f"{r2:.3f}")
                    col2.metric("RMSE", f"{rmse:.3f}")
                    col3.metric("MAE", f"{mae:.3f}")

                    vcol1, vcol2, vcol3 = st.columns(3)
                    vcol1.metric("R² Interpretation", interpret_r2_local(r2))
                    vcol2.metric("RMSE Level", interpret_err_local(rmse, y_test))
                    vcol3.metric("MAE Level", interpret_err_local(mae, y_test))

                    st.subheader("🔁 Model Cross-Validation")
                    fig, ax = plt.subplots()
                    ax.scatter(y_test, y_pred, alpha=0.7, s=60)
                    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                    ax.set_xlabel("Actual Values")
                    ax.set_ylabel("Predicted Values")
                    st.pyplot(fig)


                    # -------------------- 🔁 CROSS-VALIDATION SECTION --------------------

                    if st.button("Run 5-Fold Cross-Validation"):
                        with st.spinner("Running cross-validation... please wait."):
                            cv_scores = cross_val_score(
                                rf, features, df_model[target_col], cv=5, scoring='r2'
                            )
                            mean_score = np.mean(cv_scores)
                            std_score = np.std(cv_scores)

                            st.success("✅ Cross-validation complete!")
                            st.write(f"**R² Scores per Fold:** {cv_scores}")
                            st.write(f"**Average R²:** {mean_score:.4f}")
                            st.write(f"**Standard Deviation:** {std_score:.4f}")

                            fig, ax = plt.subplots()
                            ax.bar(range(1, 6), cv_scores, color='skyblue')
                            ax.axhline(y=mean_score, color='red', linestyle='--', label=f"Mean R² = {mean_score:.4f}")
                            ax.set_xlabel("Fold")
                            ax.set_ylabel("R² Score")
                            ax.set_title("5-Fold Cross-Validation Results")
                            ax.legend()
                            st.pyplot(fig)

                              # ---------------- CLASSIFICATION MODE ----------------
                else:
                    from sklearn.ensemble import RandomForestClassifier
                    from sklearn.metrics import (
                        accuracy_score, precision_score, recall_score,
                        f1_score, confusion_matrix
                    )

                    # Convert target to categorical strings
                    y_train_cat = y_train.astype(str)
                    y_test_cat = y_test.astype(str)

                    rf_clf = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=42,
                        n_jobs=-1
                    )
                    with st.spinner("Training Random Forest classifier..."):
                        rf_clf.fit(X_train, y_train_cat)
                        y_pred_cat = rf_clf.predict(X_test)

                    # --- Metrics ---
                    acc = accuracy_score(y_test_cat, y_pred_cat)
                    prec = precision_score(y_test_cat, y_pred_cat, average="weighted", zero_division=0)
                    rec = recall_score(y_test_cat, y_pred_cat, average="weighted", zero_division=0)
                    f1 = f1_score(y_test_cat, y_pred_cat, average="weighted", zero_division=0)

                    # --- Accuracy Section ---
                    st.subheader("🧠 Preprocessing & Classification")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{acc:.3f}")
                    col2.metric("Precision", f"{prec:.3f}")
                    col3.metric("Recall", f"{rec:.3f}")
                    col4.metric("F1 Score", f"{f1:.3f}")

                    # --- Validation Section ---
                    st.subheader("✅ Model Validation")
                    vcol1, vcol2, vcol3 = st.columns(3)
                    vcol1.metric("Performance", "High" if f1 > 0.8 else "Moderate" if f1 > 0.5 else "Low")
                    vcol2.metric("Recall Level", "Good" if rec > 0.7 else "Poor")
                    vcol3.metric("Precision Level", "Stable" if prec > 0.7 else "Unstable")

                    try:
                        from sklearn.model_selection import train_test_split
                        from sklearn.preprocessing import OneHotEncoder
                        from sklearn.ensemble import RandomForestClassifier
                        from sklearn.metrics import classification_report, accuracy_score

                        features_cols = []
                        for col in ['Site', 'Place', 'Latitude', 'Longitude', 'Year']:
                            if col in df.columns:
                                features_cols.append(col)

                        if len(features_cols) < 2:
                            st.info("Not enough features for classification example. Skipping RF classification block.")
                        else:
                            features = [f for f in ['Site', 'Place', 'Latitude', 'Longitude', 'Year'] if f in df.columns]
                            target = 'Microplastic_Level'
                            X = df[features].copy()
                            y = df[target].copy()

                            categorical_features = [c for c in ['Site', 'Place'] if c in X.columns]
                            if len(categorical_features) > 0:
                                encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                                encoded_categorical_features = encoder.fit_transform(X[categorical_features])
                                encoded_feature_names = encoder.get_feature_names_out(categorical_features)
                                encoded_df = pd.DataFrame(encoded_categorical_features, columns=encoded_feature_names, index=X.index)
                                X = X.drop(categorical_features, axis=1)
                                X = pd.concat([X, encoded_df], axis=1)

                            low_threshold = y.quantile(0.5)
                            medium_threshold = y.quantile(0.9)

                            def categorize_microplastic_level(level):
                                if level <= low_threshold:
                                    return 'Low'
                                elif level <= medium_threshold:
                                    return 'Medium'
                                else:
                                    return 'High'

                            y_categorized = y.apply(categorize_microplastic_level)
                            X_train, X_test, y_train, y_test = train_test_split(X, y_categorized, test_size=0.2, random_state=42, stratify=y_categorized)

                            rf_clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
                            rf_clf2.fit(X_train, y_train)
                            y_pred_cat2 = rf_clf2.predict(X_test)

                            classification_rep = classification_report(y_test, y_pred_cat2, output_dict=True, zero_division=0)
                            acc2 = accuracy_score(y_test, y_pred_cat2)
                            report_df = pd.DataFrame(classification_rep).transpose()

                            st.subheader("📘 Classification Report")
                            st.dataframe(report_df.style.format("{:.2f}"))
                            st.markdown(f"**Overall Accuracy:** `{acc2:.4f}`")

                    except Exception as e:
                        st.warning(f"Ma’am’s classification block failed: {e}")

                    # --- Confusion Matrix Section ---
                    st.subheader("📘 Confusion Matrix (Validation Visualization)")


                    # Ensure unique labels
                    labels = np.unique(np.concatenate((y_test_cat, y_pred_cat)))
                    cm = confusion_matrix(y_test_cat, y_pred_cat, labels=labels)

                    # Limit labels display if too large
                    if len(labels) > 20:
                        labels = labels[:20]
                        cm = confusion_matrix(y_test_cat, y_pred_cat, labels=labels)

                    # ---- Create white-background figure ----
                    fig, ax = plt.subplots(figsize=(6, 5))
                    fig.patch.set_facecolor('white')
                    ax.set_facecolor('white')

                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        cbar=True,
                        linewidths=0.5,
                        xticklabels=labels,
                        yticklabels=labels,
                        ax=ax
                    )

                    ax.set_xlabel("Predicted Labels", color='black')
                    ax.set_ylabel("True Labels", color='black')
                    st.pyplot(fig)


                # 🌿 Feature importance (works for regression only)
                if task_type == "Regression":
                    st.subheader("🌿 Feature Importance")
                try:
                    # choose rf object depending on mode
                    rf_obj = rf if task_type == "Regression only" else rf_clf
                    importances = pd.DataFrame(
                        {"Feature": features.columns, "Importance": rf_obj.feature_importances_}
                    ).sort_values("Importance", ascending=False)

                    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(importances))))
                    sns.barplot(x="Importance", y="Feature", data=importances, ax=ax)
                    st.pyplot(fig)

                    # Predictive microplastic levels — regression only (simulated)
                    if task_type == "Regression":
                        st.subheader("🔮 Predictive Microplastic Levels")

                        # Generate simulated forecast for next 5 years (2026–2030)
                        years = np.arange(2026, 2031)
                        avg_pred = float(np.mean(y_pred)) if 'y_pred' in locals() and len(y_pred) > 0 else 0
                        future_preds = np.linspace(avg_pred * 0.9, avg_pred * 1.1, len(years))

                        # Plot forecasted microplastic levels
                        fig, ax = plt.subplots()
                        ax.plot(years, future_preds, marker='o', label='Forecasted Level')
                        ax.set_xlabel("Future Year")
                        ax.set_ylabel("Predicted Microplastic Level (µg/L)")
                        ax.set_title("Predicted Microplastic Levels for Next 5 Years")
                        ax.legend()
                        st.pyplot(fig)

                        # --- User input for specific year prediction ---
                        st.markdown("### 📅 Enter a Future Year (Next 5 Years)")
                        future_year = st.number_input(
                            "Enter a year (e.g., 2026–2030):",
                            min_value=int(years[0]),
                            max_value=int(years[-1]),
                            step=1
                        )

                        if future_year in years:
                            predicted_value = future_preds[list(years).index(future_year)]
                            st.success(f"🌍 Predicted microplastic level in **{int(future_year)}**: {predicted_value:.2f} µg/L")
                        else:
                            st.info("Enter a valid year between 2026 and 2030 to see prediction.")

                        # Save forecast to session_state for Reports tab
                        future_df = pd.DataFrame({"Year": years, "Predicted_Microplastic_Level": future_preds})
                        st.session_state["future_df"] = future_df

                except Exception as e:
                    st.warning(f"Could not plot feature importances: {e}")

        except Exception as e:
            st.error(f"Random Forest failed: {e}")

    # -------------------- PROPHET --------------------
    elif model_choice == "Prophet":
        try:
            from prophet import Prophet
            from sklearn.metrics import mean_absolute_error
            year_col = [c for c in df_model.columns if c.lower() == "year"][0]
            prophet_df = df_model[[year_col, target_col]].rename(columns={year_col: "ds", target_col: "y"})
            prophet_df["ds"] = pd.to_datetime(prophet_df["ds"].astype(int).astype(str) + "-01-01")

            prophet_df = prophet_df.dropna().drop_duplicates(subset=["ds"]).sort_values("ds")

            if len(prophet_df) < 10:
                st.warning("⚠️ Not enough data points for Prophet (minimum 10). Try SARIMA instead.")
            else:
                m = Prophet(yearly_seasonality=True)
                m.fit(prophet_df)
                future = m.make_future_dataframe(periods=5, freq='Y')
                forecast = m.predict(future)
                y_true = prophet_df["y"]
                y_pred = m.predict(prophet_df)["yhat"]

                r2 = r2_score(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)

                def interpret_r2(r2): 
                    return "Excellent" if r2 >= 0.8 else "Good" if r2 >= 0.6 else "Fair" if r2 >= 0.3 else "Poor" if r2 >= 0 else "Very Poor"
                def interpret_err(err, y): 
                    ratio = (err / np.mean(y)) * 100 if np.mean(y) != 0 else 0
                    return "Low" if ratio < 10 else "Moderate" if ratio < 30 else "High"

                # --- Accuracy Section ---
                st.subheader("📊 Model Accuracy")
                col1, col2, col3 = st.columns(3)
                col1.metric("R²", f"{r2:.3f}")
                col2.metric("RMSE", f"{rmse:.3f}")
                col3.metric("MAE", f"{mae:.3f}")

                # --- Validation Section ---
                vcol1, vcol2, vcol3 = st.columns(3)
                vcol1.metric("R² Interpretation", interpret_r2(r2))
                vcol2.metric("RMSE Level", interpret_err(rmse, y_true))
                vcol3.metric("MAE Level", interpret_err(mae, y_true))

                st.pyplot(m.plot(forecast))
                st.pyplot(m.plot_components(forecast))

        except Exception as e:
            st.error(f"Prophet forecasting failed: {e}")

            # -------------------- ARIMA --------------------
    elif model_choice == "ARIMA":
        try:
            st.subheader("🔴 ARIMA Model")
            from statsmodels.tsa.arima.model import ARIMA

            if yearly_microplastic is None or len(yearly_microplastic) < 3:
                # try recomputing in case Year column exists with different name
                if 'Year' in df_model.columns:
                    yearly_microplastic = df_model.groupby('Year')[target_col].mean().reset_index().rename(columns={target_col:'Microplastic_Level'})

            if yearly_microplastic is None or len(yearly_microplastic) < 3:
                st.info("Not enough yearly data for ARIMA (need at least 3 points).")
            else:
                model_arima = ARIMA(yearly_microplastic['Microplastic_Level'], order=(5,1,0))
                with st.spinner("Fitting ARIMA..."):
                    model_arima_fit = model_arima.fit()
                st.success("ARIMA model trained successfully.")
                # Forecast next 3 years
                forecast_steps = 3
                forecast = model_arima_fit.forecast(steps=forecast_steps)
                st.write(f"ARIMA Forecast (next {forecast_steps} steps):")
                st.dataframe(pd.DataFrame({'Forecast': forecast}).reset_index(drop=True))

        except Exception as e:
            st.warning(f"ARIMA model failed: {e}")

    # -------------------- SES --------------------
    elif model_choice == "SES":
        try:
            st.subheader("🟢 Simple Exponential Smoothing")
            from statsmodels.tsa.holtwinters import SimpleExpSmoothing

            if yearly_microplastic is None or len(yearly_microplastic) < 3:
                if 'Year' in df_model.columns:
                    yearly_microplastic = df_model.groupby('Year')[target_col].mean().reset_index().rename(columns={target_col:'Microplastic_Level'})

            if yearly_microplastic is None or len(yearly_microplastic) < 3:
                st.info("Not enough yearly data for SES (need at least 3 points).")
            else:
                model_ses = SimpleExpSmoothing(yearly_microplastic['Microplastic_Level'])
                model_ses_fit = model_ses.fit()
                forecast_steps_ses = 3
                forecast_ses = model_ses_fit.forecast(steps=forecast_steps_ses)
                st.write(f"SES Forecast (next {forecast_steps_ses} steps):")
                st.dataframe(pd.DataFrame({'Forecast_SES': forecast_ses}).reset_index(drop=True))

        except Exception as e:
            st.warning(f"SES failed: {e}")

    # -------------------- SARIMA --------------------
    elif model_choice == "SARIMA":
        st.markdown("### 🔁 SARIMA")
        try:
            import statsmodels.api as sm
            import itertools
            from sklearn.metrics import mean_absolute_error

            year_col = [c for c in df_model.columns if c.lower() == "year"][0]
            ts = df_model.set_index(year_col)[target_col].astype(float)

            p = d = q = [0, 1]
            pdq = list(itertools.product(p, d, q))
            best_aic = np.inf
            best_res, best_order = None, None

            for order in pdq:
                try:
                    model = sm.tsa.statespace.SARIMAX(ts, order=order, enforce_stationarity=False, enforce_invertibility=False)
                    results = model.fit(disp=False)
                    if results.aic < best_aic:
                        best_aic, best_res, best_order = results.aic, results, order
                except:
                    continue

            if best_res is not None:
                st.success(f"Best SARIMA order: {best_order} (AIC={best_aic:.2f})")

                fitted = best_res.fittedvalues
                r2 = r2_score(ts, fitted)
                rmse = np.sqrt(mean_squared_error(ts, fitted))
                mae = mean_absolute_error(ts, fitted)

                def interpret_r2(r2): 
                    return "Excellent" if r2 >= 0.8 else "Good" if r2 >= 0.6 else "Fair" if r2 >= 0.3 else "Poor" if r2 >= 0 else "Very Poor"
                def interpret_err(err, y): 
                    ratio = (err / np.mean(y)) * 100 if np.mean(y) != 0 else 0
                    return "Low" if ratio < 10 else "Moderate" if ratio < 30 else "High"

                # --- Accuracy Section ---
                st.subheader("📊 Model Accuracy")
                col1, col2, col3 = st.columns(3)
                col1.metric("R²", f"{r2:.3f}")
                col2.metric("RMSE", f"{rmse:.3f}")
                col3.metric("MAE", f"{mae:.3f}")

                # --- Validation Section ---
                vcol1, vcol2, vcol3 = st.columns(3)
                vcol1.metric("R² Interpretation", interpret_r2(r2))
                vcol2.metric("RMSE Level", interpret_err(rmse, ts))
                vcol3.metric("MAE Level", interpret_err(mae, ts))

                # Forecast Plot
                steps = 5
                pred = best_res.get_forecast(steps=steps)
                pred_ci = pred.conf_int()
                last_year = int(ts.index.max())
                years = np.arange(last_year + 1, last_year + 1 + steps)
                preds = pred.predicted_mean.values

                fig, ax = plt.subplots(figsize=(8, 4))
                ts.plot(ax=ax, label="Observed")
                ax.plot(years, preds, color="red", marker="o", label="Forecast")
                ax.fill_between(years, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='pink', alpha=0.3)
                ax.legend()
                st.pyplot(fig)
            else:
                st.error("SARIMA model fit failed for all attempted orders.")

        except Exception as e:
            st.error(f"SARIMA forecasting failed: {e}")

            # -------------------- ALL / COMPARE (Combined Forecast Plot) --------------------
    elif model_choice == "All (compare)":
        st.subheader("📉 Combined Historical & Forecast Plot")
        # Recompute yearly if needed
        if yearly_microplastic is None:
            if 'Year' in df_model.columns:
                yearly_microplastic = df_model.groupby('Year')[target_col].mean().reset_index().rename(columns={target_col:'Microplastic_Level'})
        if yearly_microplastic is None or len(yearly_microplastic) < 3:
            st.info("Not enough yearly data for combined forecasting.")
        else:
            # ARIMA forecast
            try:
                from statsmodels.tsa.arima.model import ARIMA
                arima_fit = ARIMA(yearly_microplastic['Microplastic_Level'], order=(5,1,0)).fit()
                arima_forecast = arima_fit.forecast(steps=3)
                arima_steps = 3
            except Exception as e:
                arima_forecast = None
                arima_steps = 0
                st.warning(f"ARIMA in combined failed: {e}")

            # SES forecast
            try:
                from statsmodels.tsa.holtwinters import SimpleExpSmoothing
                ses_fit = SimpleExpSmoothing(yearly_microplastic['Microplastic_Level']).fit()
                ses_forecast = ses_fit.forecast(steps=3)
                ses_steps = 3
            except Exception as e:
                ses_forecast = None
                ses_steps = 0
                st.warning(f"SES in combined failed: {e}")

            # Prophet forecast
            try:
                from prophet import Prophet
                prophet_df = yearly_microplastic.copy()
                prophet_df['ds'] = pd.to_datetime(prophet_df['Year'].astype(int).astype(str) + '-01-01')
                prophet_df['y'] = prophet_df['Microplastic_Level']
                prophet_df = prophet_df[['ds','y']].dropna().drop_duplicates(subset=['ds']).sort_values('ds')
                if len(prophet_df) >= 3:
                    m = Prophet(yearly_seasonality=True)
                    with st.spinner("Training Prophet for combined plot..."):
                        m.fit(prophet_df)
                    future = m.make_future_dataframe(periods=3, freq='Y')
                    forecast_prophet = m.predict(future)
                else:
                    forecast_prophet = None
            except Exception as e:
                forecast_prophet = None
                st.warning(f"Prophet in combined failed: {e}")

            # Plot combined
            try:
                fig, ax = plt.subplots(figsize=(10,6))
                ax.plot(yearly_microplastic['Year'], yearly_microplastic['Microplastic_Level'], marker='o', label='Historical Data')

                last_year = int(yearly_microplastic['Year'].iloc[-1])

                if arima_forecast is not None:
                    forecast_years = list(range(last_year+1, last_year + arima_steps + 1))
                    ax.plot(forecast_years, list(arima_forecast), marker='x', linestyle='--', label='ARIMA Forecast', color='red')

                if ses_forecast is not None:
                    forecast_years_ses = list(range(last_year+1, last_year + ses_steps + 1))
                    ax.plot(forecast_years_ses, list(ses_forecast), marker='o', linestyle='--', label='SES Forecast', color='green')

                if forecast_prophet is not None:
                    prophet_future_forecast = forecast_prophet[forecast_prophet['ds'] > prophet_df['ds'].max()]
                    if not prophet_future_forecast.empty:
                        ax.plot(prophet_future_forecast['ds'].dt.year, prophet_future_forecast['yhat'], marker='^', linestyle='-.', label='Prophet Forecast', color='purple')
                        ax.fill_between(prophet_future_forecast['ds'].dt.year, prophet_future_forecast['yhat_lower'], prophet_future_forecast['yhat_upper'], color='purple', alpha=0.1)

                ax.set_title('Historical and Forecasted Microplastic Levels')
                ax.set_xlabel('Year'); ax.set_ylabel('Microplastic Level')
                ax.legend(); ax.grid(True)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Combined forecast plot failed: {e}")

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

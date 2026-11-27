"""
UNEP Food Waste Economic Loss Predictor
Streamlit app for CDS 4003 final project (Stage 7 - Deployment).

The model is rebuilt at start-up from the cleaned data and the settings chosen in
notebooks/03_modeling.ipynb (models/model_info.json), so no pickle file is needed
and the app works with any scikit-learn version.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "cleaned_food_waste.csv"
INFO_FILE = ROOT / "models" / "model_info.json"
IMPORTANCE_FILE = ROOT / "models" / "feature_importance.csv"

TARGET = "Economic Loss (Million $)"
NUM_FEATURES = ["Year", "Total Waste (Tons)", "Avg Waste per Capita (Kg)", "Household Waste (%)"]
CAT_FEATURES = ["Country", "Food Category"]

st.set_page_config(page_title="UNEP Food Waste Loss Predictor", page_icon="🌍", layout="wide")


# ---------- Load data and model ----------
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_FILE)
    # The model requires Total Waste (see 03_modeling.ipynb, section 5.1)
    return data.dropna(subset=["Total Waste (Tons)"]).reset_index(drop=True)


@st.cache_data
def load_info():
    if INFO_FILE.exists():
        return json.loads(INFO_FILE.read_text())
    return {"model_name": "Ridge (tuned linear)", "params": {"alpha": 10}}


def make_regressor(info):
    name, params = info["model_name"], info.get("params", {})
    if name.startswith("Ridge"):
        return Ridge(**params)
    if name == "Random Forest":
        return RandomForestRegressor(random_state=42, n_jobs=-1, **params)
    if name == "Gradient Boosting":
        return GradientBoostingRegressor(random_state=42, **params)
    return LinearRegression()


@st.cache_resource
def train_model(data, info):
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])
    model = Pipeline([("pre", preprocessor), ("model", make_regressor(info))])
    X = data[NUM_FEATURES + CAT_FEATURES]
    y = data[TARGET]
    # Same training split as the notebook, so the app uses the evaluated model
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    return model


df = load_data()
info = load_info()
model = train_model(df, info)

countries = sorted(df["Country"].unique())
categories = sorted(df["Food Category"].unique())
waste_min, waste_max = float(df["Total Waste (Tons)"].min()), float(df["Total Waste (Tons)"].max())

# ---------- Header ----------
st.title("🌍 Food Waste Economic Loss Predictor")
st.markdown(
    "Built for the **UN Environment Programme (UNEP)**. Countries usually report food waste in **tons**; "
    "this tool estimates the **economic loss in million US$** so UNEP can see where waste costs the most "
    "and focus action there first."
)

col1, col2, col3 = st.columns(3)
col1.metric("Model", info.get("model_name", "—"))
col2.metric("Test R²", f"{info.get('test_r2', float('nan')):.3f}")
col3.metric("Typical error (MAE)", f"{info.get('test_mae', float('nan')):,.0f} M$")

tab_predict, tab_compare, tab_about = st.tabs(["🔮 Predict a loss", "📊 Where does waste cost the most?", "ℹ️ About the model"])

# ---------- Tab 1: Prediction ----------
with tab_predict:
    st.subheader("Enter a country's food waste record")
    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        country = c1.selectbox("Country", countries)
        category = c2.selectbox("Food category", categories)
        year = c1.slider("Year", 2018, 2024, 2024)
        total_waste = c2.number_input("Total waste (tons) — required", min_value=0.0,
                                      max_value=1_000_000.0, value=25_000.0, step=500.0)
        per_capita = c1.number_input("Average waste per capita (kg)", min_value=0.0,
                                     max_value=1_000.0, value=float(df["Avg Waste per Capita (Kg)"].median()))
        household = c2.slider("Household waste (%)", 0.0, 100.0, float(round(df["Household Waste (%)"].median(), 1)))
        submitted = st.form_submit_button("Predict economic loss")

    if submitted:
        # Input validation
        if total_waste <= 0:
            st.error("Total waste must be greater than 0 tons. The model needs the measured tonnage to estimate the loss.")
        else:
            if not (waste_min <= total_waste <= waste_max):
                st.warning(
                    f"⚠️ {total_waste:,.0f} tons is outside the range the model was trained on "
                    f"({waste_min:,.0f}–{waste_max:,.0f} tons). The estimate is an extrapolation and less reliable."
                )
            if per_capita == 0:
                st.info("Average waste per capita is 0 — check this value. It has very little effect on the prediction.")

            row = pd.DataFrame([{
                "Year": year, "Total Waste (Tons)": total_waste, "Avg Waste per Capita (Kg)": per_capita,
                "Household Waste (%)": household, "Country": country, "Food Category": category,
            }])
            prediction = float(model.predict(row)[0])
            mae = info.get("test_mae")

            st.success(f"### Estimated economic loss: **{prediction:,.0f} million US$**")
            if mae:
                st.caption(f"Typical error on unseen data: ± {mae:,.0f} million US$ (mean absolute error).")

            # Explanation of the prediction
            st.markdown("#### Why this number?")
            per_ton = prediction / total_waste
            st.markdown(
                f"- The model estimates about **{per_ton:.2f} million US$ per ton** of waste for this record.\n"
                f"- **Total waste is by far the most important input**: in testing, almost all of the model's "
                f"accuracy came from it. Country, category, year and household share change the result only slightly.\n"
                f"- Compared with the average **{category}** record in the data "
                f"({df.loc[df['Food Category'] == category, TARGET].mean():,.0f} M$), this prediction is "
                f"{'higher' if prediction > df.loc[df['Food Category'] == category, TARGET].mean() else 'lower'}."
            )

            # Sensitivity: what if waste were reduced by 10%? (links to the SMART objective)
            reduced = row.copy()
            reduced["Total Waste (Tons)"] = total_waste * 0.9
            saving = prediction - float(model.predict(reduced)[0])
            st.markdown(
                f"#### What if waste fell by 10%?\nCutting this waste by 10% would save an estimated "
                f"**{saving:,.0f} million US$** — the kind of reduction UNEP aims for in the highest-loss areas."
            )

    if IMPORTANCE_FILE.exists():
        with st.expander("Feature importance (from the test set)"):
            importance = pd.read_csv(IMPORTANCE_FILE).set_index("Feature")
            st.bar_chart(importance)

# ---------- Tab 2: Comparison ----------
with tab_compare:
    st.subheader("Predicted economic loss by country and food category")
    st.markdown("Predictions for every record in the data, added up. Use this to choose where UNEP should act first.")

    preds = df.copy()
    preds["Predicted Loss (M$)"] = model.predict(preds[NUM_FEATURES + CAT_FEATURES])

    year_filter = st.multiselect("Years", sorted(preds["Year"].unique()), default=sorted(preds["Year"].unique()))
    view = preds[preds["Year"].isin(year_filter)]

    if view.empty:
        st.warning("Select at least one year.")
    else:
        c1, c2 = st.columns(2)
        c1.markdown("**By country**")
        c1.bar_chart(view.groupby("Country")["Predicted Loss (M$)"].sum().sort_values(ascending=False))
        c2.markdown("**By food category**")
        c2.bar_chart(view.groupby("Food Category")["Predicted Loss (M$)"].sum().sort_values(ascending=False))

        st.markdown("**Top 5 country + category combinations (UNEP priority list)**")
        top5 = (view.groupby(["Country", "Food Category"])["Predicted Loss (M$)"].sum()
                  .sort_values(ascending=False).head(5).round(0).reset_index())
        st.dataframe(top5, use_container_width=True, hide_index=True)

        selected = st.selectbox("Look inside one country", countries)
        st.bar_chart(view[view["Country"] == selected]
                     .groupby("Food Category")["Predicted Loss (M$)"].sum().sort_values(ascending=False))

# ---------- Tab 3: About ----------
with tab_about:
    st.subheader("About this model")
    st.markdown(f"""
- **Data:** Messy Global Food Wastage Dataset (Kaggle, CC0), cleaned in `notebooks/01_data_cleaning.ipynb`.
  {len(df):,} records from {df['Country'].nunique()} countries, 2018–2024.
- **Model:** {info.get('model_name', '—')} with settings {info.get('params', {})}, chosen in `notebooks/03_modeling.ipynb`
  after comparing a baseline, Linear Regression, Random Forest and Gradient Boosting with cross-validation and tuning.
- **Performance on unseen test data:** R² = {info.get('test_r2', float('nan')):.3f}, MAE = {info.get('test_mae', float('nan')):,.0f} M$.

**Limitations**
- The dataset's values cannot be verified as real-world statistics.
- Only {df['Country'].nunique()} countries are covered; predictions for other countries are not supported.
- Economic loss in this data is almost proportional to tonnage, so the model mainly converts tons into dollars.
- Total waste must be known; the model was not trained on records without it.
""")

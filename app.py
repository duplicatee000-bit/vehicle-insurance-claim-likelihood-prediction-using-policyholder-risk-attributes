import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vehicle Insurance Claim Predictor",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Generate Synthetic Dataset ─────────────────────────────────────────────────
@st.cache_data
def generate_data(n=5000):
    np.random.seed(42)
    age          = np.random.randint(18, 75, n)
    gender       = np.random.choice(["Male", "Female"], n)
    vehicle_age  = np.random.randint(0, 20, n)
    annual_km    = np.random.randint(5000, 80000, n)
    prev_claims  = np.random.poisson(0.8, n)
    credit_score = np.random.randint(300, 850, n)
    vehicle_type = np.random.choice(["Sedan", "SUV", "Truck", "Hatchback", "Sports"], n)
    region       = np.random.choice(["Urban", "Suburban", "Rural"], n, p=[0.5, 0.3, 0.2])
    coverage     = np.random.choice(["Basic", "Standard", "Premium"], n)
    safe_driver  = np.random.choice([0, 1], n, p=[0.35, 0.65])
    premium      = np.random.randint(8000, 80000, n)

    risk = (
        0.30 * (age < 25).astype(int) +
        0.20 * (vehicle_age > 10).astype(int) +
        0.35 * (prev_claims > 1).astype(int) +
        0.15 * (annual_km > 50000).astype(int) +
        0.25 * (credit_score < 500).astype(int) +
        0.10 * (np.array(vehicle_type) == "Sports").astype(int) +
        0.15 * (np.array(region) == "Urban").astype(int) -
        0.20 * safe_driver +
        0.10 * (np.array(gender) == "Male").astype(int) +
        np.random.normal(0, 0.1, n)
    )
    claim = (risk > 0.55).astype(int)

    return pd.DataFrame({
        "Age": age, "Gender": gender, "VehicleAge": vehicle_age,
        "AnnualKM": annual_km, "PreviousClaims": prev_claims,
        "CreditScore": credit_score, "VehicleType": vehicle_type,
        "Region": region, "CoverageType": coverage,
        "SafeDriverDiscount": safe_driver, "AnnualPremium": premium,
        "Claim": claim
    })


# ── Encoding Map ───────────────────────────────────────────────────────────────
ENC = {
    "Gender":       {"Male": 1, "Female": 0},
    "VehicleType":  {"Sedan": 0, "SUV": 1, "Truck": 2, "Hatchback": 3, "Sports": 4},
    "Region":       {"Urban": 2, "Suburban": 1, "Rural": 0},
    "CoverageType": {"Basic": 0, "Standard": 1, "Premium": 2},
}

def encode_df(df):
    d = df.copy()
    for col, mapping in ENC.items():
        d[col] = d[col].map(mapping)
    return d


# ── Train Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def train_models(_df):
    dfe = encode_df(_df)
    X = dfe.drop("Claim", axis=1)
    y = dfe["Claim"]

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    }

    results = {}
    for name, mdl in models.items():
        mdl.fit(X_tr, y_tr)
        y_pred = mdl.predict(X_te)
        y_prob = mdl.predict_proba(X_te)[:, 1]
        results[name] = {
            "model":    mdl,
            "accuracy": accuracy_score(y_te, y_pred),
            "auc":      roc_auc_score(y_te, y_prob),
            "cm":       confusion_matrix(y_te, y_pred),
            "y_te":     y_te.values,
            "y_prob":   y_prob,
        }
    return results, scaler, list(X.columns)


# ── Plotly dark layout helper ──────────────────────────────────────────────────
def dlayout(h=350):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        margin=dict(l=10, r=10, t=35, b=10),
        height=h,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA & MODELS
# ══════════════════════════════════════════════════════════════════════════════
df = generate_data()
model_results, scaler, feature_cols = train_models(df)


# ── Header ────────────────────────────────────────────────────────────────────
st.title("Vehicle Insurance Claim Predictor")
st.caption("Predict the likelihood of an insurance claim using policyholder risk attributes.")
st.divider()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    chosen_model = st.selectbox("Model", list(model_results.keys()))
    st.divider()

    st.header("Policyholder Details")
    age          = st.slider("Age", 18, 74, 35)
    gender       = st.selectbox("Gender", ["Male", "Female"])
    vehicle_age  = st.slider("Vehicle Age (years)", 0, 20, 5)
    annual_km    = st.number_input("Annual KM Driven", min_value=5000, max_value=80000, value=20000, step=1000)
    prev_claims  = st.number_input("Previous Claims", min_value=0, max_value=10, value=0)
    credit_score = st.slider("Credit Score", 300, 850, 650)
    vehicle_type = st.selectbox("Vehicle Type", ["Sedan", "SUV", "Truck", "Hatchback", "Sports"])
    region       = st.selectbox("Region", ["Urban", "Suburban", "Rural"])
    coverage     = st.selectbox("Coverage Type", ["Basic", "Standard", "Premium"])
    safe_driver  = st.selectbox("Safe Driver Discount", ["Yes", "No"])
    annual_prem  = st.number_input("Annual Premium (Rs.)", min_value=8000, max_value=80000, value=25000, step=1000)
    st.divider()
    predict_btn  = st.button("Predict Claim Risk", use_container_width=True, type="primary")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Model Analytics", "Data Insights", "Dataset"])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if predict_btn:
        row = np.array([[
            age,
            ENC["Gender"][gender],
            vehicle_age,
            annual_km,
            prev_claims,
            credit_score,
            ENC["VehicleType"][vehicle_type],
            ENC["Region"][region],
            ENC["CoverageType"][coverage],
            1 if safe_driver == "Yes" else 0,
            annual_prem,
        ]], dtype=float)

        row_sc = scaler.transform(row)
        res    = model_results[chosen_model]
        prob   = float(res["model"].predict_proba(row_sc)[0][1])

        # Risk tier
        if prob >= 0.60:
            tier   = "HIGH RISK"
            color  = "#ef4444"
        elif prob >= 0.30:
            tier   = "MEDIUM RISK"
            color  = "#f59e0b"
        else:
            tier   = "LOW RISK"
            color  = "#10b981"

        col_g, col_d = st.columns(2, gap="large")

        with col_g:
            st.subheader("Claim Probability Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                number={"suffix": "%", "font": {"size": 42}},
                gauge={
                    "axis":        {"range": [0, 100]},
                    "bar":         {"color": color, "thickness": 0.3},
                    "bgcolor":     "lightgrey",
                    "steps": [
                        {"range": [0,  30], "color": "rgba(16,185,129,0.15)"},
                        {"range": [30, 60], "color": "rgba(245,158,11,0.15)"},
                        {"range": [60,100], "color": "rgba(239,68,68,0.15)"},
                    ],
                }
            ))
            fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=280, margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown(f"### Risk Level: :{'red' if prob>=0.6 else ('orange' if prob>=0.3 else 'green')}[{tier}]")

        with col_d:
            st.subheader("Risk Factor Breakdown")
            factors = {
                "Young Driver (<25)":       0.30 if age < 25 else 0.0,
                "Old Vehicle (>10 yrs)":    0.20 if vehicle_age > 10 else 0.0,
                "Multiple Claims History":  0.35 if prev_claims > 1 else 0.0,
                "High Mileage (>50k km)":   0.15 if annual_km > 50000 else 0.0,
                "Low Credit Score (<500)":  0.25 if credit_score < 500 else 0.0,
                "Sports Vehicle":           0.10 if vehicle_type == "Sports" else 0.0,
                "Urban Region":             0.15 if region == "Urban" else 0.0,
                "Safe Driver (reduces)":   -0.20 if safe_driver == "Yes" else 0.0,
            }
            labels = list(factors.keys())
            values = list(factors.values())
            bar_colors = ["#ef4444" if v > 0 else "#10b981" for v in values]

            fig_bar = go.Figure(go.Bar(
                x=values, y=labels, orientation="h",
                marker_color=bar_colors,
                text=[f"{v:+.2f}" for v in values],
                textposition="outside",
            ))
            fig_bar.update_layout(
                **dlayout(340),
                xaxis=dict(showgrid=False, zeroline=True, zerolinecolor="#555", range=[-0.35, 0.5]),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Claim Probability", f"{prob:.1%}")
        m2.metric("Outcome", "Claim Likely" if prob >= 0.5 else "Claim Unlikely")
        m3.metric("Model Accuracy",  f"{res['accuracy']:.1%}")
        m4.metric("ROC-AUC",         f"{res['auc']:.3f}")

        st.divider()
        st.subheader("Insurer Recommendations")
        if prob >= 0.60:
            st.error("High risk profile detected.")
            st.markdown("""
- Consider a premium loading of **20 to 35%**
- Request full driving history and vehicle inspection
- Schedule a mandatory annual policy review
- Consider telematics or usage-based insurance
            """)
        elif prob >= 0.30:
            st.warning("Moderate risk profile.")
            st.markdown("""
- Apply a **5 to 15%** loading on base premium
- Monitor claim frequency over next policy year
- Offer safe-driver incentive program enrollment
            """)
        else:
            st.success("Low risk profile — eligible for preferred pricing.")
            st.markdown("""
- Apply No-Claim Bonus at renewal
- Enroll in loyalty rewards program
- Cross-sell opportunity: health or life insurance bundle
            """)

    else:
        st.info("Fill in policyholder details in the sidebar and click **Predict Claim Risk**.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — MODEL ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Model Performance Comparison")
    c1, c2, c3 = st.columns(3)
    for col, (name, res) in zip([c1, c2, c3], model_results.items()):
        col.metric(name, f"{res['accuracy']:.1%} Accuracy", f"AUC: {res['auc']:.3f}")

    st.divider()
    col_roc, col_cm = st.columns(2)

    with col_roc:
        st.markdown("**ROC Curves**")
        fig_roc = go.Figure()
        palette = ["#3b82f6", "#10b981", "#f59e0b"]
        for (name, res), c in zip(model_results.items(), palette):
            fpr, tpr, _ = roc_curve(res["y_te"], res["y_prob"])
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f"{name.split()[0]} AUC={res['auc']:.3f}",
                line=dict(color=c, width=2.5)
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1],
            line=dict(dash="dot", color="grey", width=1),
            showlegend=False
        ))
        fig_roc.update_layout(
            **dlayout(360),
            xaxis=dict(title="False Positive Rate", showgrid=False),
            yaxis=dict(title="True Positive Rate",  showgrid=False),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_cm:
        st.markdown(f"**Confusion Matrix — {chosen_model}**")
        cm = model_results[chosen_model]["cm"]
        fig_cm = px.imshow(
            cm, text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual"),
            x=["No Claim", "Claim"],
            y=["No Claim", "Claim"],
        )
        fig_cm.update_layout(**dlayout(360), coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

    st.divider()
    st.markdown("**Feature Importance — Gradient Boosting**")
    gb_model    = model_results["Gradient Boosting"]["model"]
    importances = gb_model.feature_importances_
    feat_df     = pd.DataFrame({"Feature": feature_cols, "Importance": importances}).sort_values("Importance")
    fig_fi = go.Figure(go.Bar(
        x=feat_df["Importance"], y=feat_df["Feature"], orientation="h",
        marker_color="#3b82f6",
        text=[f"{v:.3f}" for v in feat_df["Importance"]],
        textposition="outside",
    ))
    fig_fi.update_layout(
        **dlayout(380),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig_fi, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — DATA INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Dataset Overview")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total Records", f"{len(df):,}")
    d2.metric("Claim Rate",    f"{df['Claim'].mean():.1%}")
    d3.metric("Average Age",   f"{df['Age'].mean():.0f} yrs")
    d4.metric("Avg Premium",   f"Rs.{df['AnnualPremium'].mean():,.0f}")

    st.divider()
    r1, r2 = st.columns(2)

    with r1:
        st.markdown("**Age Distribution by Claim**")
        fig_age = go.Figure()
        for val, lbl, clr in [(0, "No Claim", "#10b981"), (1, "Claim", "#ef4444")]:
            fig_age.add_trace(go.Histogram(
                x=df[df["Claim"] == val]["Age"],
                name=lbl, opacity=0.7, marker_color=clr, nbinsx=25,
            ))
        fig_age.update_layout(**dlayout(300), barmode="overlay",
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
        st.plotly_chart(fig_age, use_container_width=True)

    with r2:
        st.markdown("**Claim Rate by Vehicle Type**")
        vt = df.groupby("VehicleType")["Claim"].mean().reset_index().sort_values("Claim", ascending=False)
        fig_vt = px.bar(
            vt, x="VehicleType", y="Claim",
            color="Claim", color_continuous_scale=["green", "red"],
            labels={"Claim": "Claim Rate", "VehicleType": ""},
        )
        fig_vt.update_layout(**dlayout(300), coloraxis_showscale=False,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False, tickformat=".0%"))
        st.plotly_chart(fig_vt, use_container_width=True)

    r3, r4 = st.columns(2)

    with r3:
        st.markdown("**Claim Rate by Credit Score Band**")
        df["CreditBand"] = pd.cut(df["CreditScore"], bins=6)
        cs = df.groupby("CreditBand", observed=True)["Claim"].mean().reset_index()
        cs["Band"] = cs["CreditBand"].astype(str)
        fig_cs = go.Figure(go.Bar(
            x=cs["Band"], y=cs["Claim"],
            marker_color="#3b82f6",
            text=[f"{v:.0%}" for v in cs["Claim"]],
            textposition="outside",
        ))
        fig_cs.update_layout(**dlayout(300),
            xaxis=dict(showgrid=False, tickangle=-20),
            yaxis=dict(showgrid=False, tickformat=".0%"))
        st.plotly_chart(fig_cs, use_container_width=True)

    with r4:
        st.markdown("**Claim Rate by Region**")
        rg = df.groupby("Region")["Claim"].mean().reset_index()
        fig_rg = go.Figure(go.Pie(
            labels=rg["Region"], values=rg["Claim"],
            hole=0.5,
            marker=dict(colors=["#3b82f6", "#10b981", "#f59e0b"]),
        ))
        fig_rg.update_layout(**dlayout(300))
        st.plotly_chart(fig_rg, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — DATASET
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Sample Dataset")
    n_rows    = st.slider("Rows to display", 10, 200, 50)
    sample_df = df.sample(n_rows, random_state=1).reset_index(drop=True)
    st.dataframe(sample_df, use_container_width=True, height=420)
    st.divider()
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Full Dataset (CSV)",
        data=csv_data,
        file_name="vehicle_insurance_claims.csv",
        mime="text/csv",
    )

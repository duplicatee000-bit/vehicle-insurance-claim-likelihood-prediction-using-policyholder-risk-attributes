import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InsureIQ · Claim Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

:root {
    --bg: #0a0e1a;
    --surface: #111827;
    --surface2: #1a2235;
    --accent: #3b82f6;
    --accent2: #10b981;
    --accent3: #f59e0b;
    --danger: #ef4444;
    --text: #e2e8f0;
    --muted: #64748b;
    --border: #1e293b;
}

html, body, .stApp {
    background-color: var(--bg) !important;
    font-family: 'Space Grotesk', sans-serif;
    color: var(--text);
}

.main .block-container { padding: 2rem 2.5rem; max-width: 1400px; }

/* Hero Header */
.hero-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    border: 1px solid #1e3a8a;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(59,130,246,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #34d399, #60a5fa);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite;
    margin: 0;
    line-height: 1.1;
}
@keyframes shimmer { to { background-position: 200% center; } }
.hero-sub { color: #94a3b8; font-size: 1.05rem; margin-top: 0.5rem; font-weight: 400; }

/* Metric Cards */
.metric-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-bottom: 1.5rem; }
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover { transform: translateY(-2px); border-color: var(--accent); }
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0;
    width: 100%; height: 3px;
}
.metric-card.blue::after  { background: var(--accent); }
.metric-card.green::after { background: var(--accent2); }
.metric-card.amber::after { background: var(--accent3); }
.metric-card.red::after   { background: var(--danger); }

.metric-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); margin-bottom: 0.4rem; }
.metric-value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 700; color: var(--text); }
.metric-sub   { font-size: 0.8rem; color: var(--muted); margin-top: 0.2rem; }

/* Risk Gauge */
.risk-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.risk-label { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.15em; color: var(--muted); margin-bottom: 1rem; }

/* Section Headers */
.section-head {
    display: flex; align-items: center; gap: 0.75rem;
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem; font-weight: 700;
    color: var(--text);
    margin: 1.5rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}
.section-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent);
}

/* Prediction Badge */
.pred-badge {
    display: inline-block;
    padding: 0.6rem 2rem;
    border-radius: 50px;
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    margin: 1rem 0;
}
.pred-high { background: rgba(239,68,68,0.15); color: #fca5a5; border: 2px solid rgba(239,68,68,0.4); }
.pred-low  { background: rgba(16,185,129,0.15); color: #6ee7b7; border: 2px solid rgba(16,185,129,0.4); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label { color: var(--muted) !important; font-size: 0.82rem !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface2);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    color: var(--muted);
    font-weight: 500;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: white !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.8rem;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
    cursor: pointer;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover { background: linear-gradient(135deg, #3b82f6, #2563eb); transform: translateY(-1px); box-shadow: 0 4px 15px rgba(59,130,246,0.3); }

/* Inputs */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}
.stSlider > div > div > div { color: var(--accent) !important; }

hr { border-color: var(--border) !important; }
p, li { color: var(--text); }
h3 { font-family: 'Syne', sans-serif; color: var(--text); }
.stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Data Generation ────────────────────────────────────────────────────────────
@st.cache_data
def generate_dataset(n=5000):
    np.random.seed(42)
    age         = np.random.randint(18, 75, n)
    gender      = np.random.choice(["Male", "Female"], n)
    vehicle_age = np.random.randint(0, 20, n)
    annual_km   = np.random.randint(5000, 80000, n)
    prev_claims = np.random.poisson(0.8, n)
    credit_score= np.random.randint(300, 850, n)
    vehicle_type= np.random.choice(["Sedan", "SUV", "Truck", "Hatchback", "Sports"], n)
    region      = np.random.choice(["Urban", "Suburban", "Rural"], n, p=[0.5, 0.3, 0.2])
    coverage    = np.random.choice(["Basic", "Standard", "Premium"], n)
    safe_driver = np.random.choice([0, 1], n, p=[0.35, 0.65])
    annual_premium = np.random.randint(8000, 80000, n)

    # Probabilistic label
    risk = (
        0.3 * (age < 25).astype(int) +
        0.2 * (vehicle_age > 10).astype(int) +
        0.35 * (prev_claims > 1).astype(int) +
        0.15 * (annual_km > 50000).astype(int) +
        0.25 * (credit_score < 500).astype(int) +
        0.1  * (vehicle_type == "Sports").astype(int) +
        0.15 * (region == "Urban").astype(int) -
        0.2  * safe_driver +
        0.1  * (gender == "Male").astype(int) +
        np.random.normal(0, 0.1, n)
    )
    claim = (risk > 0.55).astype(int)

    return pd.DataFrame({
        "Age": age, "Gender": gender, "VehicleAge": vehicle_age,
        "AnnualKM": annual_km, "PreviousClaims": prev_claims,
        "CreditScore": credit_score, "VehicleType": vehicle_type,
        "Region": region, "CoverageType": coverage,
        "SafeDriverDiscount": safe_driver, "AnnualPremium": annual_premium,
        "Claim": claim
    })


# ─── Model Training ─────────────────────────────────────────────────────────────
@st.cache_resource
def train_models(df):
    le = LabelEncoder()
    dfe = df.copy()
    for col in ["Gender", "VehicleType", "Region", "CoverageType"]:
        dfe[col] = le.fit_transform(dfe[col])

    X = dfe.drop("Claim", axis=1)
    y = dfe["Claim"]
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42),
        "Random Forest":     RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    }
    results = {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)
        y_prob = m.predict_proba(X_te)[:, 1]
        results[name] = {
            "model": m,
            "accuracy": accuracy_score(y_te, y_pred),
            "auc":      roc_auc_score(y_te, y_prob),
            "cm":       confusion_matrix(y_te, y_pred),
            "y_te":     y_te, "y_prob": y_prob,
            "report":   classification_report(y_te, y_pred, output_dict=True),
        }
    return results, scaler, X.columns.tolist()


# ─── Plotly theme helper ────────────────────────────────────────────────────────
LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk", color="#94a3b8"),
    margin=dict(l=10, r=10, t=40, b=10),
)

# ══════════════════════════════════════════════════════════════════════════════
#                              MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
df = generate_dataset()
model_results, scaler, feature_cols = train_models(df)

# ─── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <p class="hero-title">🛡️ InsureIQ</p>
  <p class="hero-sub">Vehicle Insurance Claim Likelihood Prediction · Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    st.markdown("**Select Model**")
    chosen_model = st.selectbox("", list(model_results.keys()), label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**📋 Policyholder Attributes**")

    age           = st.slider("Age", 18, 74, 35)
    gender        = st.selectbox("Gender", ["Male", "Female"])
    vehicle_age   = st.slider("Vehicle Age (years)", 0, 20, 5)
    annual_km     = st.number_input("Annual KM Driven", 5000, 80000, 20000, step=1000)
    prev_claims   = st.number_input("Previous Claims", 0, 10, 0)
    credit_score  = st.slider("Credit Score", 300, 850, 650)
    vehicle_type  = st.selectbox("Vehicle Type", ["Sedan", "SUV", "Truck", "Hatchback", "Sports"])
    region        = st.selectbox("Region", ["Urban", "Suburban", "Rural"])
    coverage      = st.selectbox("Coverage Type", ["Basic", "Standard", "Premium"])
    safe_driver   = st.selectbox("Safe Driver Discount", ["Yes", "No"])
    annual_prem   = st.number_input("Annual Premium (₹)", 8000, 80000, 25000, step=1000)

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Claim Risk")


# ─── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "📊 Model Analytics", "📈 Data Insights", "📋 Dataset"])

# ════════════════════════════════
#  TAB 1 · PREDICTION
# ════════════════════════════════
with tab1:
    if predict_btn:
        res  = model_results[chosen_model]
        m    = res["model"]

        enc_map = {
            "Gender":       {"Male": 1, "Female": 0},
            "VehicleType":  {"Sedan": 3, "SUV": 4, "Truck": 5, "Hatchback": 1, "Sports": 2},
            "Region":       {"Urban": 2, "Suburban": 1, "Rural": 0},
            "CoverageType": {"Basic": 0, "Standard": 2, "Premium": 1},
        }
        row = np.array([[
            age,
            enc_map["Gender"][gender],
            vehicle_age,
            annual_km,
            prev_claims,
            credit_score,
            enc_map["VehicleType"][vehicle_type],
            enc_map["Region"][region],
            enc_map["CoverageType"][coverage],
            1 if safe_driver == "Yes" else 0,
            annual_prem,
        ]])
        row_sc  = scaler.transform(row)
        prob    = m.predict_proba(row_sc)[0][1]
        pred    = int(prob >= 0.5)

        # Risk tier
        if prob < 0.3:   tier, tier_col = "LOW RISK",    "#10b981"
        elif prob < 0.6: tier, tier_col = "MEDIUM RISK", "#f59e0b"
        else:            tier, tier_col = "HIGH RISK",   "#ef4444"

        # Layout
        col_g, col_d = st.columns([1, 1])

        with col_g:
            st.markdown('<div class="risk-container">', unsafe_allow_html=True)
            st.markdown('<p class="risk-label">Claim Probability</p>', unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                number={"suffix": "%", "font": {"size": 40, "color": "#e2e8f0", "family": "Syne"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#334155", "tickfont": {"color": "#64748b"}},
                    "bar": {"color": tier_col, "thickness": 0.25},
                    "bgcolor": "#1a2235",
                    "bordercolor": "#1e293b",
                    "steps": [
                        {"range": [0, 30],  "color": "rgba(16,185,129,0.15)"},
                        {"range": [30, 60], "color": "rgba(245,158,11,0.15)"},
                        {"range": [60, 100],"color": "rgba(239,68,68,0.15)"},
                    ],
                    "threshold": {"line": {"color": tier_col, "width": 4}, "value": prob * 100},
                }
            ))
            fig_gauge.update_layout(**LAYOUT, height=280)
            st.plotly_chart(fig_gauge, use_container_width=True)

            badge_cls = "pred-high" if pred == 1 else "pred-low"
            badge_txt = "⚠️ CLAIM LIKELY" if pred == 1 else "✅ CLAIM UNLIKELY"
            st.markdown(f'<div style="text-align:center"><span class="pred-badge {badge_cls}">{badge_txt}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<p style="text-align:center;color:{tier_col};font-weight:600;font-size:1.1rem">{tier}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_d:
            st.markdown('<div class="section-head"><span class="section-dot"></span>Risk Factor Breakdown</div>', unsafe_allow_html=True)
            factors = {
                "Young Driver (< 25)":        0.3 if age < 25 else 0,
                "Old Vehicle (> 10 yrs)":     0.2 if vehicle_age > 10 else 0,
                "Multiple Claims History":    0.35 if prev_claims > 1 else 0,
                "High Annual Mileage":        0.15 if annual_km > 50000 else 0,
                "Low Credit Score (< 500)":   0.25 if credit_score < 500 else 0,
                "Sports Vehicle":             0.1  if vehicle_type == "Sports" else 0,
                "Urban Region":               0.15 if region == "Urban" else 0,
                "Safe Driver Discount":      -0.2 if safe_driver == "Yes" else 0,
            }
            labels = list(factors.keys())
            vals   = list(factors.values())
            colors = ["#ef4444" if v > 0 else "#10b981" for v in vals]

            fig_bar = go.Figure(go.Bar(
                x=vals, y=labels, orientation="h",
                marker_color=colors,
                text=[f"{v:+.2f}" for v in vals],
                textposition="outside", textfont={"color": "#94a3b8"},
            ))
            fig_bar.update_layout(
                **LAYOUT, height=320,
                xaxis=dict(showgrid=False, zeroline=True, zerolinecolor="#334155", color="#64748b"),
                yaxis=dict(showgrid=False, color="#94a3b8"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Summary metrics
        st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
        cols4 = st.columns(4)
        with cols4[0]:
            st.metric("Claim Probability", f"{prob:.1%}")
        with cols4[1]:
            st.metric("Model Used", chosen_model.split()[0])
        with cols4[2]:
            st.metric("Model Accuracy", f"{res['accuracy']:.1%}")
        with cols4[3]:
            st.metric("ROC-AUC Score", f"{res['auc']:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Recommendation
        st.markdown('<div class="section-head"><span class="section-dot"></span>Insurer Recommendations</div>', unsafe_allow_html=True)
        if prob >= 0.6:
            rec = [
                "🔴 **High risk profile detected** — consider a loading of 20–35% on base premium.",
                "📋 Request additional documentation: driving history & vehicle inspection report.",
                "🔁 Schedule mandatory annual review for this policyholder.",
                "💡 Offer telematics/usage-based insurance to monitor real-time driving behavior.",
            ]
        elif prob >= 0.3:
            rec = [
                "🟡 **Moderate risk** — standard premium with a 5–15% loading is advisable.",
                "📊 Monitor claim frequency over the next policy year.",
                "💡 Offer safe-driver discount incentive program to reduce risk.",
            ]
        else:
            rec = [
                "🟢 **Low risk profile** — eligible for preferred pricing.",
                "⭐ Consider loyalty rewards and no-claim bonus for next renewal.",
                "💡 Cross-sell opportunity: life or health insurance bundle.",
            ]
        for r in rec:
            st.markdown(r)
    else:
        st.markdown("""
        <div style="text-align:center; padding:5rem 2rem; color:#334155;">
            <p style="font-size:4rem; margin:0">🛡️</p>
            <p style="font-size:1.5rem; color:#475569; font-family:'Syne',sans-serif; font-weight:700; margin-top:1rem">
                Fill in the policyholder details and click <b style="color:#3b82f6">Predict Claim Risk</b>
            </p>
            <p style="color:#334155; margin-top:0.5rem">Configure attributes in the left sidebar to get started.</p>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════
#  TAB 2 · MODEL ANALYTICS
# ════════════════════════════════
with tab2:
    st.markdown('<div class="section-head"><span class="section-dot"></span>Model Performance Comparison</div>', unsafe_allow_html=True)

    # KPI cards
    cols3 = st.columns(3)
    pal = ["blue", "green", "amber"]
    for i, (name, res) in enumerate(model_results.items()):
        with cols3[i]:
            st.markdown(f"""
            <div class="metric-card {pal[i]}">
              <div class="metric-label">{name}</div>
              <div class="metric-value">{res['accuracy']:.1%}</div>
              <div class="metric-sub">AUC: {res['auc']:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_roc, col_cm = st.columns(2)

    with col_roc:
        st.markdown("**ROC Curves**")
        fig_roc = go.Figure()
        colors_roc = ["#3b82f6", "#10b981", "#f59e0b"]
        for (name, res), c in zip(model_results.items(), colors_roc):
            fpr, tpr, _ = roc_curve(res["y_te"], res["y_prob"])
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name.split()[0]} (AUC={res['auc']:.3f})",
                                         line=dict(color=c, width=2)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash="dot", color="#334155"), showlegend=False))
        fig_roc.update_layout(**LAYOUT, height=350,
            xaxis=dict(title="FPR", showgrid=False, color="#64748b"),
            yaxis=dict(title="TPR", showgrid=False, color="#64748b"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_cm:
        st.markdown(f"**Confusion Matrix — {chosen_model}**")
        cm = model_results[chosen_model]["cm"]
        fig_cm = px.imshow(cm, text_auto=True,
                           labels=dict(x="Predicted", y="Actual"),
                           color_continuous_scale=[[0,"#1a2235"],[1,"#3b82f6"]],
                           x=["No Claim","Claim"], y=["No Claim","Claim"])
        fig_cm.update_layout(**LAYOUT, height=350,
            coloraxis_showscale=False,
            xaxis=dict(color="#64748b"), yaxis=dict(color="#64748b"),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # Feature Importance
    st.markdown('<div class="section-head"><span class="section-dot"></span>Feature Importance</div>', unsafe_allow_html=True)
    best_model = model_results["Gradient Boosting"]["model"]
    importances = best_model.feature_importances_
    feat_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances}).sort_values("Importance")
    fig_fi = go.Figure(go.Bar(
        x=feat_df["Importance"], y=feat_df["Feature"], orientation="h",
        marker=dict(color=feat_df["Importance"],
                    colorscale=[[0,"#1e3a5f"],[0.5,"#3b82f6"],[1,"#60a5fa"]]),
    ))
    fig_fi.update_layout(**LAYOUT, height=350,
        xaxis=dict(showgrid=False, color="#64748b"),
        yaxis=dict(showgrid=False, color="#94a3b8"),
    )
    st.plotly_chart(fig_fi, use_container_width=True)


# ════════════════════════════════
#  TAB 3 · DATA INSIGHTS
# ════════════════════════════════
with tab3:
    st.markdown('<div class="section-head"><span class="section-dot"></span>Dataset Overview</div>', unsafe_allow_html=True)
    cols4 = st.columns(4)
    stats = [
        ("Total Records", f"{len(df):,}", "blue"),
        ("Claim Rate",    f"{df['Claim'].mean():.1%}", "red"),
        ("Avg Age",       f"{df['Age'].mean():.0f} yrs", "green"),
        ("Avg Premium",   f"₹{df['AnnualPremium'].mean():,.0f}", "amber"),
    ]
    for col, (label, val, color) in zip(cols4, stats):
        with col:
            st.markdown(f"""
            <div class="metric-card {color}">
              <div class="metric-label">{label}</div>
              <div class="metric-value" style="font-size:1.6rem">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Age Distribution by Claim**")
        fig_age = go.Figure()
        for claim_val, name, color in [(0, "No Claim", "#10b981"), (1, "Claim", "#ef4444")]:
            fig_age.add_trace(go.Histogram(
                x=df[df["Claim"] == claim_val]["Age"],
                name=name, opacity=0.7, marker_color=color,
                nbinsx=25,
            ))
        fig_age.update_layout(**LAYOUT, height=300, barmode="overlay",
            xaxis=dict(showgrid=False, color="#64748b"),
            yaxis=dict(showgrid=False, color="#64748b"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
        )
        st.plotly_chart(fig_age, use_container_width=True)

    with c2:
        st.markdown("**Claim Rate by Vehicle Type**")
        vt_rate = df.groupby("VehicleType")["Claim"].mean().reset_index()
        fig_vt = px.bar(vt_rate, x="VehicleType", y="Claim",
                        color="Claim", color_continuous_scale=["#10b981","#ef4444"],
                        labels={"Claim": "Claim Rate", "VehicleType": ""})
        fig_vt.update_layout(**LAYOUT, height=300,
            coloraxis_showscale=False,
            xaxis=dict(showgrid=False, color="#64748b"),
            yaxis=dict(showgrid=False, tickformat=".0%", color="#64748b"),
        )
        st.plotly_chart(fig_vt, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("**Credit Score vs Claim Probability**")
        bins = pd.cut(df["CreditScore"], bins=10)
        cs_rate = df.groupby(bins, observed=True)["Claim"].mean().reset_index()
        cs_rate["CreditScore"] = cs_rate["CreditScore"].astype(str)
        fig_cs = go.Figure(go.Scatter(
            x=cs_rate["CreditScore"], y=cs_rate["Claim"],
            fill="tozeroy", fillcolor="rgba(59,130,246,0.15)",
            line=dict(color="#3b82f6", width=2.5),
        ))
        fig_cs.update_layout(**LAYOUT, height=280,
            xaxis=dict(showgrid=False, color="#64748b", showticklabels=False),
            yaxis=dict(showgrid=False, tickformat=".0%", color="#64748b"),
        )
        st.plotly_chart(fig_cs, use_container_width=True)

    with c4:
        st.markdown("**Claim Rate by Region**")
        reg_rate = df.groupby("Region")["Claim"].mean().reset_index()
        fig_reg = go.Figure(go.Pie(
            labels=reg_rate["Region"], values=reg_rate["Claim"],
            hole=0.6,
            marker=dict(colors=["#3b82f6","#10b981","#f59e0b"]),
            textfont=dict(color="#e2e8f0"),
        ))
        fig_reg.update_layout(**LAYOUT, height=280,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
        )
        st.plotly_chart(fig_reg, use_container_width=True)


# ════════════════════════════════
#  TAB 4 · DATASET
# ════════════════════════════════
with tab4:
    st.markdown('<div class="section-head"><span class="section-dot"></span>Sample Dataset</div>', unsafe_allow_html=True)
    show_n = st.slider("Rows to display", 10, 200, 50)
    sample = df.sample(show_n, random_state=1).reset_index(drop=True)
    st.dataframe(sample, use_container_width=True, height=420)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Full Dataset (CSV)", csv, "insurance_claims.csv", "text/csv")

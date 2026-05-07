"""Streamlit UI for the Zimbabwe Student Dropout predictor.

Run with:
    streamlit run app.py --server.port 8502
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ARTIFACTS = Path(__file__).parent / "artifacts"
DATA_PATH = Path(r"C:/Users/nicky/Documents/zimbabwe_student_dropout (1).csv")

MODEL_FILES = {
    "Logistic Regression": "logreg.joblib",
    "KNN":                 "knn.joblib",
    "Naive Bayes":         "nb.joblib",
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models…")
def load_models() -> dict:
    return {name: joblib.load(ARTIFACTS / fn) for name, fn in MODEL_FILES.items()}


@st.cache_data
def load_schema() -> dict:
    with open(ARTIFACTS / "feature_columns.json") as f:
        return json.load(f)


@st.cache_data
def load_metrics() -> dict:
    with open(ARTIFACTS / "metrics.json") as f:
        return json.load(f)


@st.cache_data
def load_sample_rows(n: int = 200) -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH, nrows=10_000)
    return df.sample(n, random_state=0).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Page config + theming
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Zimbabwe Student Dropout Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* Hide default streamlit chrome that we replace */
    [data-testid="stSidebarCollapsedControl"] { display: none; }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e40af 60%, #2563eb 100%);
        padding: 2.2rem 2rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 4px 24px rgba(15, 23, 42, 0.15);
    }
    .hero h1 { color: #fff; margin: 0; font-size: 2.1rem; line-height: 1.15; }
    .hero p  { color: #cbd5e1; margin: 0.4rem 0 0; font-size: 1rem; max-width: 60ch; }
    .hero .badges { margin-top: 0.9rem; }
    .badge {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.2);
        color: #e0e7ff;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        font-size: 0.78rem;
        margin-right: 0.4rem;
    }

    /* Section header bar */
    .sec {
        margin: 1.2rem 0 0.6rem;
        padding: 0.5rem 0.85rem;
        background: #f1f5f9;
        border-left: 4px solid #3b82f6;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 700;
        color: #1e3a8a;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* Result cards */
    .res {
        padding: 1.6rem 1.2rem;
        border-radius: 14px;
        text-align: center;
        margin-top: 0.8rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }
    .res-stay { background: linear-gradient(135deg,#dcfce7,#bbf7d0); border: 2px solid #16a34a; }
    .res-drop { background: linear-gradient(135deg,#fee2e2,#fecaca); border: 2px solid #dc2626; }
    .res .model { font-size: 0.82rem; opacity: 0.7; text-transform: uppercase;
                  letter-spacing: 0.06em; font-weight: 600; }
    .res .label { font-size: 1.35rem; font-weight: 700; margin: 0.45rem 0 0.3rem; color: #0f172a; }
    .res .pct   { font-size: 2.6rem; font-weight: 800; color: #0f172a; line-height: 1; }
    .res .sub   { font-size: 0.78rem; opacity: 0.65; margin-top: 0.3rem; }
    .res-single .pct { font-size: 3.6rem; }
    .res-single .label { font-size: 1.7rem; }

    /* Probability bar */
    .pbar-wrap {
        background: #f1f5f9; border-radius: 999px; height: 18px;
        margin: 0.8rem 0; overflow: hidden; position: relative;
    }
    .pbar-fill {
        height: 100%; border-radius: 999px;
        background: linear-gradient(90deg,#16a34a 0%,#facc15 50%,#dc2626 100%);
    }
    .pbar-marker {
        position: absolute; top: -4px; width: 4px; height: 26px;
        background: #0f172a; border-radius: 2px;
    }
    .pbar-scale {
        display: flex; justify-content: space-between;
        font-size: 0.72rem; color: #64748b; margin-top: -0.3rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>Zimbabwe Student Dropout Predictor</h1>
  <p>Three classifiers trained on 523,574 student records estimate a student's
     dropout risk from 14 demographic, academic, financial and environmental signals.</p>
  <div class="badges">
    <span class="badge">Logistic Regression</span>
    <span class="badge">K-Nearest Neighbours</span>
    <span class="badge">Naive Bayes</span>
    <span class="badge">scikit-learn pipelines</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Artifact check
missing = [fn for fn in MODEL_FILES.values() if not (ARTIFACTS / fn).exists()]
if missing or not (ARTIFACTS / "feature_columns.json").exists():
    st.error(
        "**Trained models not found.**  \n"
        "Open `train_models.ipynb` in VS Code and *Run All* cells first."
    )
    st.stop()

models      = load_models()
schema      = load_schema()
metrics     = load_metrics()
sample_rows = load_sample_rows()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_predict, tab_models, tab_data = st.tabs(
    ["  Predict  ", "  Models & metrics  ", "  About the data  "]
)


# ============================================================
# TAB 1 — Predict
# ============================================================
with tab_predict:
    # Top control row: model picker + autofill button
    ctrl_left, ctrl_right = st.columns([3, 1])
    with ctrl_left:
        model_choice = st.radio(
            "Model",
            options=["Compare all 3", *MODEL_FILES.keys()],
            horizontal=True,
            label_visibility="collapsed",
        )
    with ctrl_right:
        if not sample_rows.empty:
            if st.button("Autofill from real row", use_container_width=True):
                st.session_state["_autofill"] = sample_rows.sample(1).iloc[0].to_dict()
                st.rerun()

    auto = st.session_state.get("_autofill", {})

    def num(label, key, *, step=1.0, integer=False):
        b = schema["numeric"][key]
        d = auto.get(key, b["median"])
        if d is None or (isinstance(d, float) and pd.isna(d)):
            d = b["median"]
        if integer:
            return st.number_input(
                label, min_value=int(b["min"]), max_value=int(b["max"]),
                value=int(d), step=int(step), key=f"in_{key}",
            )
        return st.number_input(
            label, min_value=float(b["min"]), max_value=float(b["max"]),
            value=float(d), step=float(step), key=f"in_{key}",
        )

    def cat(label, key):
        opts = schema["categorical"][key]
        d = auto.get(key, opts[0])
        if d not in opts:
            d = opts[0]
        return st.selectbox(label, opts, index=opts.index(d), key=f"in_{key}")

    # ---------- Personal ----------
    st.markdown('<div class="sec">Personal</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: age      = num("Age (years)",  "age", integer=True)
    with c2: gender   = cat("Gender",       "gender")
    with c3: location = cat("Location",     "location")

    # ---------- Academic ----------
    st.markdown('<div class="sec">Academic background</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        gd = auto.get("previous_grade", "2.1")
        if gd not in schema["grade_labels"]:
            gd = "2.1"
        previous_grade = st.selectbox(
            "Previous grade", schema["grade_labels"],
            index=schema["grade_labels"].index(gd),
            help="Zim scale: 1 = best, F = fail",
        )
    with c2: attendance  = num("Attendance (%)",  "attendance_rate", step=1.0)
    with c3: study_hours = num("Study hrs / day", "study_hours",     step=0.1)
    with c4: lms_logins  = num("LMS logins / wk", "lms_logins",      integer=True)

    # ---------- Financial ----------
    st.markdown('<div class="sec">Financial situation</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: family_income = num("Family income (USD/wk)", "family_income", step=10.0)
    with c2: fees_paid     = cat("Fees paid",              "fees_paid")
    with c3: part_time_job = cat("Part-time job",          "part_time_job")

    # ---------- Environment ----------
    st.markdown('<div class="sec">Home & environment</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: internet       = cat("Internet access",     "internet_access")
    with c2: electricity    = cat("Electricity",         "electricity_reliability")
    with c3: transport_time = num("Transport (min)",     "transport_time", step=1.0)
    with c4: stress_level   = num("Stress level (0-10)", "stress_level",   step=0.1)

    st.markdown("&nbsp;", unsafe_allow_html=True)
    predict_clicked = st.button(
        "Predict dropout risk →", type="primary", use_container_width=True,
    )

    if predict_clicked:
        row = pd.DataFrame([{
            "age":                     age,
            "gender":                  gender,
            "location":                location,
            "family_income":           family_income,
            "internet_access":         internet,
            "electricity_reliability": electricity,
            "transport_time":          transport_time,
            "study_hours":             study_hours,
            "attendance_rate":         attendance,
            "lms_logins":              lms_logins,
            "previous_grade_ord":      schema["grade_map"][previous_grade],
            "fees_paid":               fees_paid,
            "part_time_job":           part_time_job,
            "stress_level":            stress_level,
        }])

        chosen = list(MODEL_FILES.keys()) if model_choice == "Compare all 3" else [model_choice]
        results = []
        for name in chosen:
            m = models[name]
            results.append({
                "model": name,
                "pred":  int(m.predict(row)[0]),
                "proba": float(m.predict_proba(row)[0, 1]),
            })

        st.markdown('<div class="sec">Prediction</div>', unsafe_allow_html=True)

        if model_choice == "Compare all 3":
            cols = st.columns(3)
            for col, r in zip(cols, results):
                with col:
                    cls = "res-drop" if r["pred"] == 1 else "res-stay"
                    label = "Likely to drop out" if r["pred"] == 1 else "Likely to stay"
                    st.markdown(f"""
                    <div class="res {cls}">
                      <div class="model">{r['model']}</div>
                      <div class="label">{label}</div>
                      <div class="pct">{r['proba']:.0%}</div>
                      <div class="sub">probability of dropout</div>
                    </div>
                    """, unsafe_allow_html=True)

            preds = {r["pred"] for r in results}
            avg   = sum(r["proba"] for r in results) / len(results)
            if len(preds) == 1:
                st.success(
                    f"All 3 models agree — high-confidence prediction. "
                    f"Average P(dropout) = **{avg:.1%}**."
                )
            else:
                st.warning(
                    f"Models disagree — borderline case. "
                    f"Average P(dropout) = **{avg:.1%}**."
                )
        else:
            r = results[0]
            cls = "res-drop" if r["pred"] == 1 else "res-stay"
            label = "Likely to drop out" if r["pred"] == 1 else "Likely to stay"
            st.markdown(f"""
            <div class="res res-single {cls}">
              <div class="model">{r['model']}</div>
              <div class="label">{label}</div>
              <div class="pct">{r['proba']:.1%}</div>
              <div class="sub">probability of dropout</div>
            </div>
            """, unsafe_allow_html=True)

            pct = r["proba"] * 100
            st.markdown(f"""
            <div class="pbar-wrap">
              <div class="pbar-fill" style="width:100%"></div>
              <div class="pbar-marker" style="left:calc({pct:.1f}% - 2px)"></div>
            </div>
            <div class="pbar-scale">
              <span>0% — safe</span>
              <span>50%</span>
              <span>100% — high risk</span>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Show inputs sent to model"):
            st.dataframe(
                row.T.astype(str).rename(columns={0: "value"}),
                use_container_width=True,
            )
    else:
        st.info("Fill in the form, pick a model, then press **Predict dropout risk**.")


# ============================================================
# TAB 2 — Models & metrics
# ============================================================
with tab_models:
    st.subheader("Test-set performance")
    metrics_df = pd.DataFrame(metrics).T
    c1, c2, c3 = st.columns(3)
    for col, name in zip([c1, c2, c3], MODEL_FILES.keys()):
        with col:
            m = metrics[name]
            st.metric(name, f"{m['accuracy']:.1%}", help="Accuracy on held-out test set")
            st.caption(
                f"Precision **{m['precision']:.2f}**  ·  "
                f"Recall **{m['recall']:.2f}**  ·  "
                f"F1 **{m['f1']:.2f}**"
            )

    st.markdown("##### Score breakdown")
    st.dataframe(
        metrics_df.style.format("{:.4f}").background_gradient(cmap="Blues", axis=0),
        use_container_width=True,
    )
    st.bar_chart(metrics_df)

    st.markdown("##### How each model decides")
    st.markdown("""
- **Logistic Regression** — fits a weighted linear combination of features and
  squashes it through the logistic function to a probability. Trained with
  `class_weight='balanced'` to counter the 65/35 class imbalance.
- **K-Nearest Neighbours (k = 15)** — for each new student it looks up the
  15 most similar students in a 50,000-row reference set and votes. Trained on
  a subsample because predicting on 419k stored points is too slow.
- **Gaussian Naive Bayes** — assumes each numeric feature is Gaussian per
  class and combines per-feature likelihoods under independence.
  Uses balanced priors `[0.5, 0.5]`.
""")


# ============================================================
# TAB 3 — About the data
# ============================================================
with tab_data:
    st.subheader("Dataset")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",     "523,574")
    c2.metric("Columns",  "15")
    c3.metric("Dropouts", "65%")
    c4.metric("Stayed",   "35%")

    st.caption(
        f"Source file: `{DATA_PATH.name}`  ·  "
        "Target column: `dropout` (1 = left school, 0 = stayed)."
    )

    st.markdown("##### Numeric feature ranges (training data)")
    num_summary = pd.DataFrame(schema["numeric"]).T.round(2)
    num_summary.columns = ["min", "max", "median"]
    st.dataframe(num_summary, use_container_width=True)

    st.markdown("##### Categorical features")
    rows_cat = [
        {"feature": c, "values": ", ".join(map(str, vals))}
        for c, vals in schema["categorical"].items()
    ]
    rows_cat.append({
        "feature": "previous_grade",
        "values":  ", ".join(schema["grade_labels"]) + "  (ordinal: 1 best, F worst)",
    })
    st.dataframe(pd.DataFrame(rows_cat), use_container_width=True, hide_index=True)

    st.markdown("##### Missing-value handling")
    st.markdown("""
- `family_income`, `study_hours`, `attendance_rate` — about 5% missing each;
  filled with the median by `SimpleImputer` inside the pipeline.
- `fees_paid` — about 20% missing; treated as a third category `"None"`
  (the column otherwise contains only "Full" / "Partial").
""")

    if not sample_rows.empty:
        st.markdown("##### Sample rows from the CSV")
        st.dataframe(sample_rows.head(15), use_container_width=True, hide_index=True)

"""
app.py

Streamlit front-end for the Student Placement / Career Success Predictor.
Loads the trained model + preprocessing objects saved by train_model.py
and serves live predictions through a form-based UI.

Predicts whether a student profile is more likely to land a "High Package"
or a "Standard Package" offer (split at the dataset's median salary),
since the dataset's original placement-status column carries no variation
to predict against.

Run with: streamlit run app.py
"""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Placement Package Predictor",
    page_icon=":bar_chart:",
    layout="centered",
)

# ---------------------------------------------------------------------
# Light styling so the app doesn't look like a bare default form.
# No external assets, just CSS.
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        padding: 1.4rem 1.6rem;
        border-radius: 10px;
        background: linear-gradient(120deg, #1f3b57 0%, #2f5d78 100%);
        color: white;
        margin-bottom: 1.2rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.6rem;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.85;
        font-size: 0.95rem;
    }
    .result-card {
        padding: 1.2rem 1.4rem;
        border-radius: 10px;
        border: 1px solid rgba(0,0,0,0.08);
        margin-top: 0.6rem;
    }
    .result-high {
        background-color: rgba(35, 134, 54, 0.12);
        border-left: 5px solid #238636;
    }
    .result-standard {
        background-color: rgba(180, 140, 20, 0.12);
        border-left: 5px solid #b48c14;
    }
    .section-label {
        font-size: 0.8rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    target_encoder = joblib.load("target_encoder.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    best_model_name = joblib.load("best_model_name.pkl")
    salary_median = joblib.load("salary_median.pkl")
    return (model, scaler, label_encoders, target_encoder,
            feature_cols, best_model_name, salary_median)


try:
    (model, scaler, label_encoders, target_encoder,
     feature_cols, best_model_name, salary_median) = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError:
    artifacts_loaded = False

st.markdown(
    """
    <div class="main-header">
        <h1>Placement Package Predictor</h1>
        <p>Estimate whether a student profile lines up with a high or standard salary package.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not artifacts_loaded:
    st.error(
        "No trained model found in this folder. Run train_model.py first "
        "with student_placement_career_success_dataset_2026.csv present, "
        "then restart the app."
    )
    st.stop()

with st.sidebar:
    st.subheader("About this tool")
    st.write(
        "Predicts whether a student's profile is more consistent with a "
        "High Package or a Standard Package offer, split at the median "
        f"salary in the training data (₹{salary_median:.2f} LPA)."
    )
    st.write(
        "The dataset's original placement-status field marks every record "
        "as placed, so it carries no signal on its own. Salary is used "
        "instead as the basis for the prediction target."
    )
    st.write("Models evaluated during training:")
    st.markdown("- Logistic Regression\n- Decision Tree\n- Random Forest")
    st.divider()
    st.caption(f"Active model: {best_model_name}")

tab_predict, tab_model = st.tabs(["Predict", "Model details"])

with tab_predict:
    with st.form("placement_form"):
        st.markdown('<p class="section-label">Profile</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 17, 30, 21)
            gender = st.selectbox("Gender", list(label_encoders["Gender"].classes_))
        with col2:
            college_tier = st.selectbox("College tier", list(label_encoders["College_Tier"].classes_))
            specialization = st.selectbox("Specialization", list(label_encoders["Specialization"].classes_))

        st.markdown('<p class="section-label">Academics</p>', unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        with col3:
            cgpa = st.slider("CGPA", 4.5, 10.0, 7.2, 0.05)
            aptitude_score = st.slider("Aptitude test score", 35, 100, 68)
        with col4:
            resume_score = st.slider("Resume score", 45, 100, 72)
            mock_interview_score = st.slider("Mock interview score", 35, 100, 67)

        st.markdown('<p class="section-label">Technical activity</p>', unsafe_allow_html=True)
        col5, col6, col7 = st.columns(3)
        with col5:
            dsa_problems = st.number_input("DSA problems solved", 0, 900, 250)
            leetcode_rating = st.number_input("LeetCode rating", 800, 2800, 1450)
        with col6:
            github_contributions = st.number_input("GitHub contributions", 0, 500, 120)
            hackathons = st.number_input("Hackathons participated", 0, 8, 4)
        with col7:
            internships = st.number_input("Internships", 0, 4, 1)
            certifications = st.number_input("Certifications", 0, 10, 5)

        st.markdown('<p class="section-label">Skills</p>', unsafe_allow_html=True)
        col8, col9 = st.columns(2)
        with col8:
            projects_count = st.slider("Projects completed", 1, 12, 6)
            communication_skills = st.slider("Communication skills score", 40, 100, 70)
        with col9:
            ai_ml_skill = st.slider("AI/ML skill level (1-10)", 1, 10, 5)
            system_design = st.slider("System design knowledge (1-10)", 1, 10, 5)

        submitted = st.form_submit_button("Predict package", use_container_width=True)

    if submitted:
        raw_input = {
            "Age": age,
            "Gender": gender,
            "College_Tier": college_tier,
            "Specialization": specialization,
            "CGPA": cgpa,
            "DSA_Problems_Solved": dsa_problems,
            "Internships": internships,
            "Certifications": certifications,
            "Projects_Count": projects_count,
            "Communication_Skills": communication_skills,
            "Aptitude_Test_Score": aptitude_score,
            "LeetCode_Rating": leetcode_rating,
            "GitHub_Contributions": github_contributions,
            "Hackathons_Participated": hackathons,
            "AI_ML_Skill_Level": ai_ml_skill,
            "System_Design_Knowledge": system_design,
            "Resume_Score": resume_score,
            "Mock_Interview_Score": mock_interview_score,
        }

        input_df = pd.DataFrame([raw_input])

        for col, le in label_encoders.items():
            input_df[col] = le.transform(input_df[col])

        input_df = input_df[feature_cols]
        input_scaled = scaler.transform(input_df)

        pred_encoded = model.predict(input_scaled)[0]
        pred_label = target_encoder.inverse_transform([pred_encoded])[0]

        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_scaled)[0]
            high_idx = (
                list(target_encoder.classes_).index("High Package")
                if "High Package" in target_encoder.classes_ else 0
            )

        st.divider()

        if pred_label == "High Package":
            card_class = "result-high"
            headline = "Likely a High Package profile"
        else:
            card_class = "result-standard"
            headline = "Likely a Standard Package profile"

        high_prob_html = ""
        if proba is not None:
            high_prob = proba[high_idx] * 100
            high_prob_html = f"<p style='margin:0.4rem 0 0 0; color:#374151;'>Estimated probability of High Package: <b>{high_prob:.1f}%</b></p>"

        st.markdown(
            f"""
            <div class="result-card {card_class}">
                <p class="section-label">Prediction</p>
                <h3 style="margin:0;">{headline}</h3>
                {high_prob_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if proba is not None:
            st.progress(min(max(high_prob / 100, 0.0), 1.0))

        with st.expander("View input values used for this prediction"):
            st.dataframe(input_df.assign(**{"Predicted package": pred_label}))

        st.caption(
            "AI/ML skill level, internship count, and CGPA are the strongest "
            "salary correlates in this dataset and tend to drive this prediction."
        )

with tab_model:
    st.write(
        "Three classification models were trained and compared on the same "
        "train/test split, predicting High Package vs Standard Package "
        f"(split at the median salary of ₹{salary_median:.2f} LPA). The "
        "model with the best F1-score was kept for predictions in this app."
    )
    try:
        comparison_df = pd.read_csv("model_comparison.csv")
        st.dataframe(comparison_df, use_container_width=True)
    except FileNotFoundError:
        st.write("Run train_model.py to generate model_comparison.csv.")

    st.write(f"Model currently in use: **{best_model_name}**")

    try:
        st.image("eda_plots/confusion_matrix_best_model.png", caption="Confusion matrix on the test set")
    except Exception:
        pass

st.divider()
st.caption("Placement Package Predictor — built with scikit-learn and Streamlit.")
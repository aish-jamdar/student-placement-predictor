# 🎓 Student Placement / Career Success Predictor

A machine learning project that predicts whether a student's profile lines
up with a **High Package** or **Standard Package** placement outcome,
based on academic performance, technical activity, and skill scores —
wrapped in a Streamlit web app.

**Tech stack:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit

**Dataset:** `student_placement_career_success_dataset_2026.csv` (20,000 rows, 20 columns)

---

## 📁 Project Structure

```
placement_project/
├── train_model.py       # Preprocessing + EDA + model training/comparison, saves model.pkl etc.
├── app.py                # Streamlit web app that loads the saved model and makes predictions
├── requirements.txt      # Python dependencies
├── student_placement_career_success_dataset_2026.csv   # Your real dataset (place it here)
├── eda_plots/             # EDA charts saved as PNGs (created by train_model.py)
├── model_comparison.csv   # Accuracy/Precision/Recall/F1 for all 3 models
├── generate_data.py       # No longer used — kept only as a synthetic-data fallback (see note below)
└── model.pkl, scaler.pkl, label_encoders.pkl, target_encoder.pkl, feature_cols.pkl,
    best_model_name.pkl, salary_median.pkl
    # Saved model + preprocessing objects (created by train_model.py)
```

> **`generate_data.py` is no longer part of the active pipeline.** It was
> used to fabricate a placeholder dataset before you had real data. Now
> that `train_model.py` reads `student_placement_career_success_dataset_2026.csv`
> directly, you don't need to run it — it's left in the folder only as a
> fallback if you ever want to demo the app without your real CSV.

---

## ⚠️ A note on the target column

The dataset's `Placement_Status` column is **constant** — every one of the
20,000 rows is marked as placed. A column with no variation can't be used
as a classification target; a model trained on it would just always
predict "placed" without learning anything.

To keep the same placement-prediction framing while actually using the
real signal in the data, `train_model.py` derives a new target,
**`Placement_Package`**, by splitting `Salary_LPA` at its median
(₹12.43 LPA in this dataset):

- **High Package** — salary at or above the median
- **Standard Package** — salary below the median

`Salary_LPA` itself is then dropped from the model's input features (using
it as a feature would leak the answer, since it's the source of the
label). The original `Placement_Status` column is dropped entirely since
it carries no information.

If you'd rather predict **exact salary** instead of a High/Standard split,
that's a regression task — see "Customizing" below.

---

## 🧠 How It Works

1. **Data** — Your real dataset, `student_placement_career_success_dataset_2026.csv`,
   with features like CGPA, DSA problems solved, internships, certifications,
   project count, communication skills, aptitude score, LeetCode rating,
   GitHub contributions, hackathons, AI/ML skill level, system design
   knowledge, resume score, mock interview score, gender, college tier, and
   specialization.

2. **Preprocessing & EDA** — `train_model.py`:
   - Fills missing numeric values with the median and missing categorical
     values with the mode (this dataset has no missing values, but the
     logic is there for safety if you plug in messier data later).
   - Derives the `Placement_Package` target from `Salary_LPA` (see note above).
   - Encodes categorical columns (Gender, College_Tier, Specialization) with
     `LabelEncoder`.
   - Scales numeric features with `StandardScaler`.
   - Generates EDA plots (target balance, CGPA vs package, correlation
     heatmap, package rate by internships, AI/ML skill vs package) into
     `eda_plots/`.

3. **Model training & comparison** — trains **Logistic Regression**,
   **Decision Tree**, and **Random Forest**, evaluates each with Accuracy,
   Precision, Recall, and F1-score, and automatically picks the model with
   the best F1-score as the "production" model. Results are saved to
   `model_comparison.csv` and a confusion matrix plot. On this dataset all
   three models land around 70-71% accuracy — a realistic result given the
   underlying signal.

4. **Streamlit app** — `app.py` loads the saved model and preprocessing
   objects, presents a form for entering a student's profile, and returns
   a **High Package / Standard Package** prediction along with a
   probability estimate.

---

## 🛠️ Step-by-Step: Build It Locally

### 1. Set up your environment
```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Put the dataset in the project folder
Make sure `student_placement_career_success_dataset_2026.csv` sits in the
same folder as `train_model.py` and `app.py`. (No need to run
`generate_data.py` — that was only for the earlier synthetic-data version
of this project.)

### 3. Train and compare the models
```bash
python train_model.py
```
This will:
- Print data info, missing-value counts, and classification reports for
  all three models to your terminal.
- Save EDA plots into `eda_plots/`.
- Save the best-performing model and all preprocessing objects
  (`model.pkl`, `scaler.pkl`, `label_encoders.pkl`, `target_encoder.pkl`,
  `feature_cols.pkl`, `best_model_name.pkl`, `salary_median.pkl`).

### 4. Run the Streamlit app
```bash
streamlit run app.py
```
This opens the app in your browser at `http://localhost:8501`. Fill in a
student's profile and click **Predict package** to see the result.

---

## ✏️ Customizing / Improving the Project

- **Predict salary directly instead of a High/Standard split:** swap the
  classification setup for a regression one — use `Salary_LPA` as `y`
  directly, replace the three classifiers with regressors (e.g.
  `LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`),
  and score with MAE/RMSE/R² instead of Accuracy/Precision/Recall/F1.
- **Use a different dataset:** point `DATA_PATH` in `train_model.py` to any
  other CSV and update `NUMERIC_COLS` / `CATEGORICAL_COLS` / `TARGET_COL`
  (and `SALARY_COL` / `UNUSED_COLS` if relevant) to match its column names.
- **Add more models:** e.g. `xgboost.XGBClassifier`, `SVC`, or
  `KNeighborsClassifier` — just add them to the `models` dict in
  `train_model.py`.
- **Hyperparameter tuning:** wrap the Random Forest / Decision Tree in
  `GridSearchCV` or `RandomizedSearchCV` for better accuracy.
- **Feature importance:** Random Forest and Decision Tree both expose
  `.feature_importances_` — you could add a bar chart of this to `app.py`
  or `train_model.py` to explain *why* a prediction was made.
- **Class imbalance:** if your real dataset is imbalanced, consider
  `class_weight="balanced"` in the models, or `SMOTE` from `imbalanced-learn`.

---

## 🚀 Deployment Options

### Option A — Streamlit Community Cloud (easiest, free)

1. **Push your project to GitHub.**
   ```bash
   git init
   git add .
   git commit -m "Student Placement Prediction ML app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
   > Important: Commit the trained `.pkl` files **and** the dataset CSV too
   > (or add a build step that runs `train_model.py` on deploy — see
   > "Auto-train on startup" below), otherwise the deployed app won't find
   > the model.

2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in
   with GitHub.

3. Click **"New app"**, select your repository, branch (`main`), and set
   the main file path to `app.py`.

4. Click **Deploy**. Streamlit Cloud will install `requirements.txt`
   automatically and launch your app at a public URL like:
   `https://<your-app-name>.streamlit.app`

**Auto-train on startup (optional):** if you don't want to commit `.pkl`
files to GitHub (you'll still need the dataset CSV committed), add this to
the very top of `app.py`, before `load_artifacts()` is called:
```python
import os, subprocess
if not os.path.exists("model.pkl"):
    subprocess.run(["python", "train_model.py"])
```
This trains the model fresh the first time the app boots on the server.

### Option B — Render.com

1. Push your code to GitHub (same as above).
2. On [render.com](https://render.com), create a **New Web Service** and
   connect your repo.
3. Set:
   - **Build command:** `pip install -r requirements.txt && python train_model.py`
   - **Start command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Deploy — Render gives you a public HTTPS URL.

### Option C — Docker (deploy anywhere: AWS, GCP, Azure, etc.)

Create a `Dockerfile` in the project root:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
RUN python train_model.py

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t placement-predictor .
docker run -p 8501:8501 placement-predictor
```
Then push the image to any container registry (Docker Hub, ECR, GCR) and
deploy it on your cloud platform of choice.

### Option D — Hugging Face Spaces (free, great for ML demos)

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space),
   choosing **Streamlit** as the SDK.
2. Upload all the project files (`app.py`, `train_model.py`,
   `requirements.txt`, the dataset CSV, and the `.pkl` files or the
   auto-train snippet from Option A).
3. The Space builds and launches automatically, giving you a public URL.

---

## 📌 Resume-Ready Summary

> Designed and implemented a Student Placement Prediction System using
> machine learning to estimate placement outcomes from student academic
> profiles and technical skills. Preprocessed datasets, handled missing
> values, encoded categorical features, and performed exploratory data
> analysis to extract meaningful insights. Implemented and compared
> classification algorithms — Logistic Regression, Decision Tree, and
> Random Forest — achieving optimized prediction performance. Integrated
> the trained model into a user-friendly Streamlit application for
> instant placement predictions.
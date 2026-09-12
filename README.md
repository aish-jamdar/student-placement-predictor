
# 🎓 Student Placement Package Predictor

An interactive Machine Learning web application that predicts whether a student's profile is more likely to align with a **High Package** or **Standard Package** placement outcome.

The application uses academic performance, technical activity, skills, and student profile information to generate a package-category prediction along with an estimated probability.

## 🚀 Live Demo

👉 **[Try the Student Placement Package Predictor](https://student-placement-predictor-qc91.onrender.com/)**

---

## 📌 Project Overview

The **Student Placement Package Predictor** is a Machine Learning project designed to demonstrate the complete ML workflow — from data preprocessing and exploratory data analysis to model training, evaluation, and deployment.

The project uses a Streamlit-based web interface where users can enter a student's profile and receive an instant prediction.

### The application provides:

- 🎯 High Package / Standard Package prediction
- 📊 Estimated probability of a High Package
- 🤖 Active Machine Learning model
- 📈 Model comparison results
- 🧮 Input values used for prediction
- 📉 Confusion matrix of the selected model

---

## 🧠 Machine Learning Approach

Three classification algorithms were trained and compared:

1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**

Each model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

The model with the **highest F1-score** was automatically selected as the production model used by the application.

### ML Pipeline

```text
Student Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Missing Value Handling
      │
      ▼
Categorical Encoding
      │
      ▼
Train / Test Split
      │
      ▼
Feature Scaling
      │
      ▼
┌─────────────────────────┐
│     Model Training      │
│                         │
│ Logistic Regression     │
│ Decision Tree            │
│ Random Forest            │
└─────────────────────────┘
      │
      ▼
Model Evaluation
      │
      ▼
Best F1-Score Model
      │
      ▼
Saved Model + Preprocessors
      │
      ▼
Streamlit Application
      │
      ▼
Prediction
      │
      ├── High Package
      │
      └── Standard Package
````

---

## 🎯 Prediction Target

The original dataset contains a `Placement_Status` column. However, the column has the same placement status for all records, so it does not provide meaningful variation for a classification problem.

Therefore, the project derives a new target called:

### `Placement_Package`

The target is created using the median value of `Salary_LPA`.

| Category            | Definition                                    |
| ------------------- | --------------------------------------------- |
| 🟢 High Package     | Salary is greater than or equal to the median |
| 🟡 Standard Package | Salary is below the median                    |

`Salary_LPA` is then removed from the model features to prevent **target leakage**, since it is directly used to create the target.

---

## 📊 Features Used

The model uses a combination of academic, technical, skill-based, and profile features.

### 👤 Profile

* Age
* Gender
* College Tier
* Specialization

### 📚 Academic Performance

* CGPA
* Aptitude Test Score
* Resume Score
* Mock Interview Score

### 💻 Technical Activity

* DSA Problems Solved
* LeetCode Rating
* GitHub Contributions
* Hackathons Participated
* Internships
* Certifications

### 🛠️ Skills

* Projects Completed
* Communication Skills
* AI/ML Skill Level
* System Design Knowledge

---

## 🔄 Data Preprocessing

The training pipeline performs the following preprocessing steps:

* Handles missing numerical values using the median
* Handles missing categorical values using the mode
* Encodes categorical features using `LabelEncoder`
* Scales numerical features using `StandardScaler`
* Splits the dataset into training and testing sets
* Uses stratified sampling for the train/test split

The preprocessing objects are saved along with the trained model so that the same transformations can be applied to new user inputs.

---

## 📈 Model Evaluation

The trained models are compared using:

| Metric    | Purpose                              |
| --------- | ------------------------------------ |
| Accuracy  | Overall prediction correctness       |
| Precision | Correctness of positive predictions  |
| Recall    | Ability to identify positive cases   |
| F1-score  | Balance between precision and recall |

The complete comparison is stored in:

```text
model_comparison.csv
```

The model with the best F1-score is used by the Streamlit application.

---

## 🖥️ Application

The application is built using **Streamlit**.

Users can enter:

* Academic information
* Technical activity
* Skills
* College and specialization details

The application processes the input using the saved preprocessing objects and sends it to the trained model.

### Prediction Output

The application displays:

```text
Prediction:
Likely a High Package profile

Estimated probability of High Package:
XX.X%
```

The input values used for the prediction can also be viewed inside the application.

---

## 🏗️ Project Structure

```text
student-placement-predictor/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── .python-version
│
├── model.pkl
├── scaler.pkl
├── label_encoders.pkl
├── target_encoder.pkl
├── feature_cols.pkl
├── best_model_name.pkl
├── salary_median.pkl
│
└── model_comparison.csv
```

### File Description

| File                   | Purpose                                    |
| ---------------------- | ------------------------------------------ |
| `app.py`               | Streamlit application                      |
| `train_model.py`       | Data preprocessing, EDA and model training |
| `requirements.txt`     | Python dependencies                        |
| `model.pkl`            | Trained best-performing model              |
| `scaler.pkl`           | Saved feature scaler                       |
| `label_encoders.pkl`   | Saved categorical encoders                 |
| `target_encoder.pkl`   | Target label encoder                       |
| `feature_cols.pkl`     | Feature order used during training         |
| `best_model_name.pkl`  | Name of the selected model                 |
| `salary_median.pkl`    | Median salary used for target creation     |
| `model_comparison.csv` | Model evaluation results                   |

---

## 🛠️ Tech Stack

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Joblib**
* **Matplotlib**
* **Seaborn**
* **Streamlit**
* **Render**

---

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/aish-jamdar/student-placement-predictor.git
cd student-placement-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 🚀 Deployment

The application is deployed using **Render**.

### Deployment Flow

```text
GitHub Repository
        │
        ▼
      Render
        │
        ▼
Install Dependencies
        │
        ▼
Run Streamlit
        │
        ▼
   Live Web App
```

### Render Configuration

**Build Command**

```bash
pip install -r requirements.txt
```

**Start Command**

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

### 🌐 Live Application

**[Student Placement Package Predictor](https://student-placement-predictor-qc91.onrender.com/)**

---

## ⚠️ Limitations

This project provides a **Machine Learning-based estimate** and should not be considered a guaranteed prediction of an actual placement package.

Real-world placement outcomes can depend on several factors that may not be represented in the dataset, such as:

* Company-specific requirements
* Interview performance
* Job market conditions
* Role and industry
* Communication and interpersonal factors
* Individual hiring decisions

The application is primarily intended for **educational and demonstration purposes**.

---

## 🔮 Future Improvements

Potential improvements include:

* 📌 Predicting exact salary using regression
* ⚙️ Hyperparameter tuning
* 🔁 Cross-validation
* 📊 Feature importance visualization
* 🔍 Explainable AI using SHAP
* 🤖 Additional Machine Learning algorithms
* 📚 Larger and more diverse datasets
* 💡 Personalized career recommendations
* 🔄 Automated model retraining

---

## 🎓 Learning Outcomes

This project demonstrates practical experience with:

* Data preprocessing
* Exploratory Data Analysis
* Feature engineering
* Categorical encoding
* Feature scaling
* Classification algorithms
* Model comparison
* Model evaluation
* Model serialization
* Streamlit application development
* Cloud deployment
* GitHub-based project management

---

## 👩‍💻 Author

### Aishwarya Jamdar

**Project:** Student Placement Package Predictor

🌐 **Live Demo:**
[https://student-placement-predictor-qc91.onrender.com/](https://student-placement-predictor-qc91.onrender.com/)

---

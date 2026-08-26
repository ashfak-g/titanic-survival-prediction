# 🚢 Titanic Survival Prediction AI - ML Pipeline & Web Application

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Pipeline-orange?logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green?logo=flask&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

An end-to-end Machine Learning project designed to predict passenger survival probability on the Titanic. This project elevates a basic exploratory notebook into an **intermediate-level, production-grade Machine Learning pipeline** featuring a modern **Interactive Glassmorphism Flask Web Application**.

---

## 🌟 Key Features & Improvements

- 📁 **Standard ML Architecture**: Modularized codebase following industry-standard directory layout (`data/`, `notebooks/`, `src/`, `models/`, `app/`).
- ⚙️ **Feature Engineering Pipeline**: Includes passenger title extraction (`Title`), missing value imputations grouped by title/class, and engineered features (`FamilySize`, `IsAlone`).
- 🤖 **Multi-Model Comparison & Evaluation**: Rigorous evaluation across Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting using 5-Fold Stratified Cross-Validation and ROC-AUC metrics.
- 🎨 **Modern Glassmorphism Web Interface**: Responsive Flask UI rendering real-time survival status alongside survival probability percentage bars.
- ⚡ **1-Click Startup**: Included Windows batch script (`run_app.bat`) to automate environment setup, dependency installation, and web server launch.

---

## 📂 Project Directory Structure

```text
Project_Titanic/
├── 📁 app/                          # Flask Web Application
│   ├── app.py                       # Main Flask server & prediction endpoints
│   ├── 📁 static/
│   │   └── wallpaper.jpg            # Responsive UI background image
│   └── 📁 templates/
│       └── index.html               # Glassmorphism HTML/CSS UI template
├── 📁 data/                         # Project Datasets
│   ├── 📁 raw/
│   │   ├── train.csv                # Original training dataset
│   │   ├── test.csv                 # Kaggle test dataset
│   │   └── gender_submission.csv    # Benchmark submission dataset
│   └── 📁 processed/
│       └── submission.csv           # Model output submission file
├── 📁 models/
│   └── titanic_pipeline.pkl         # Serialized Scikit-Learn ML Pipeline
├── 📁 notebooks/
│   └── Titanic_EDA_and_Modeling.ipynb # Exploratory Data Analysis Notebook
├── 📁 src/                          # Modular Python Source Code
│   ├── feature_engineering.py       # Scikit-Learn custom Transformer pipeline
│   └── train.py                     # Machine Learning training & evaluation script
├── .gitignore                       # Git ignore configuration
├── requirements.txt                 # Python dependencies
├── run_app.bat                      # 1-Click execution script for Windows
└── README.md                        # Project documentation
```

---

## 📊 Model Performance Comparison

Evaluated using **5-Fold Stratified Cross-Validation**:

| Algorithm / Model | Accuracy | F1-Score | ROC-AUC Score |
| :--- | :---: | :---: | :---: |
| **Gradient Boosting Classifier (Best)** 🏆 | **83.84%** | **0.7752** | **0.8783** |
| Random Forest Classifier | 83.50% | 0.7705 | 0.8737 |
| Decision Tree Classifier | 82.83% | 0.7642 | 0.8553 |
| Logistic Regression | 81.37% | 0.7513 | 0.8649 |

---

## 🚀 Quickstart & Execution Guide

### Option 1: 1-Click Execution (Windows Users - Recommended)
Simply double-click **`run_app.bat`** in the root directory. It will automatically set up the Python virtual environment, install requirements, train the model pipeline (if not already trained), and launch the web server in your default browser at **`http://127.0.0.1:5000/`**.

---

### Option 2: Manual Terminal Setup

#### 1. Clone & Set Up Virtual Environment:
```bash
git clone https://github.com/ashfak-g/titanic-survival-prediction.git
cd titanic-survival-prediction

# Create Virtual Environment
python -m venv .venv

# Activate Virtual Environment (Windows)
.\.venv\Scripts\activate

# On Linux/macOS:
# source .venv/bin/activate
```

#### 2. Install Dependencies:
```bash
pip install -r requirements.txt
```

#### 3. Train the Model Pipeline:
```bash
python src/train.py
```

#### 4. Run Flask Web Application:
```bash
python app/app.py
```
Open **`http://127.0.0.1:5000/`** in your browser to test predictions!

---

## 🧪 Sample Test Cases

Test the model interface with these sample passenger profiles:

### 🟢 Test Case 1 (High Survival Probability):
- **Pclass**: 1st Class
- **Sex**: Female
- **Age**: 29
- **Fare**: $100
- **Embarked**: Cherbourg (C)
- 🎯 **Result**: `Survived ✅` (~99% Probability)

### 🔴 Test Case 2 (Low Survival Probability):
- **Pclass**: 3rd Class
- **Sex**: Male
- **Age**: 30
- **Fare**: $8
- **Embarked**: Southampton (S)
- 🎯 **Result**: `Did Not Survive ❌` (~11% Probability)

---

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **Machine Learning**: Scikit-Learn, Pandas, NumPy, Joblib
- **Data Visualization**: Matplotlib, Seaborn
- **Web Framework**: Flask, HTML5, CSS3 (Glassmorphism), FontAwesome
- **Environment & Automation**: Virtualenv, Windows Batch Scripting

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.

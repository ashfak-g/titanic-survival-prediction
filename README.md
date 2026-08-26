# 🚢 Titanic Survival Prediction AI - ML Pipeline & Web Application

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Pipeline-orange?logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green?logo=flask&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

টাইটানিক জাহাজের যাত্রীদের বেঁচে থাকার সম্ভাবনা (Survival Probability) প্রেডিক্ট করার জন্য একটি এন্ড-টু-এন্ড মেশিন লার্নিং প্রজেক্ট। এই প্রজেক্টটিকে বেসিক জুপিটার নোটবুক থেকে উন্নীত করে একটি সম্পূর্ণ **ইন্টারমিডিয়েট লেভেলের প্রফেশনাল মেশিন লার্নিং পাইপলাইন** এবং **Interactive Glassmorphism Flask Web Application**-এ রূপান্তর করা হয়েছে।

---

## 🌟 নতুন আপডেট ও উন্নয়ন (Key Improvements)

- 📁 **স্ট্যান্ডার্ড প্রজেক্ট আর্কিটেকচার**: সমস্ত ফাইল ও কোড রিফ্যাক্টর করে ইন্ডাস্ট্রি স্ট্যান্ডার্ড ডিরেক্টরি লেআউটে সাজানো হয়েছে (`data/`, `notebooks/`, `src/`, `models/`, `app/`)।
- ⚙️ **মডিউলার ফিচার ইঞ্জিনিয়ারিং**: নাম থেকে `Title` এক্সট্রাকশন, বয়স ও ভাড়ার গ্রুপিং ভিত্তিক ইম্পুটেশন, এবং `FamilySize` ও `IsAlone` ফিচার যুক্ত করা হয়েছে।
- 🤖 **মাল্টি-মডেল তুলনা ও এভালুয়েশন**: Logistic Regression, Decision Tree, Random Forest, এবং Gradient Boosting মডেলগুলোর মধ্যে Stratified 5-Fold Cross Validation এবং ROC-AUC স্কোরের ভিত্তিতে সেরা মডেলটি তৈরি করা হয়েছে।
- 🎨 **Modern Glassmorphism Web App**: ইউজারদের জন্য দৃষ্টিনন্দন UI, যেখানে যাত্রীর তথ্য প্রদান করলে রিয়েল-টাইমে সারভাইভাল স্ট্যাটাস ও শতকরা কত পারসেন্ট বেঁচে থাকার সম্ভাবনা তা প্রদর্শিত হয়।
- ⚡ **১-ক্লিকে রান করার সুবিধা**: Windows ব্যবহারকারীদের জন্য `run_app.bat` ডাবল ক্লিক করেই ওয়েবাসে চালু করার সুবিধা।

---

## 📂 ডিরেক্টরি স্ট্রাকচার (Project Structure)

```text
Project_Titanic/
├── 📁 app/                          # Flask Web Application
│   ├── app.py                       # Main Flask web application server
│   ├── 📁 static/
│   │   └── wallpaper.jpg            # Responsive UI Background wallpaper
│   └── 📁 templates/
│       └── index.html               # Glassmorphism HTML/CSS UI interface
├── 📁 data/                         # Data files directory
│   ├── 📁 raw/
│   │   ├── train.csv                # Original training dataset
│   │   ├── test.csv                 # Kaggle test dataset
│   │   └── gender_submission.csv    # Benchmark submission dataset
│   └── 📁 processed/
│       └── submission.csv           # Final model output submission file
├── 📁 models/
│   └── titanic_pipeline.pkl         # Trained Scikit-Learn ML Pipeline artifact
├── 📁 notebooks/
│   └── Titanic_EDA_and_Modeling.ipynb # Complete EDA & Visualization Notebook
├── 📁 src/                          # Modular Python Source Code
│   ├── feature_engineering.py       # Custom Scikit-Learn Feature Transformer
│   └── train.py                     # Machine Learning Pipeline Training Script
├── .gitignore                       # Git ignore configuration
├── requirements.txt                 # Python project dependencies
├── run_app.bat                      # 1-Click Startup script for Windows
└── README.md                        # Documentation & setup guide
```

---

## 📊 মডেল পারফরম্যান্স (Model Evaluation Results)

5-Fold Stratified Cross-Validation ব্যবহার করে মূল্যায়ন করা ফলসমূহ:

| অ্যালগরিদম (Model Name) | একিউরেসি (Accuracy) | F1-Score | ROC-AUC Score |
| :--- | :---: | :---: | :---: |
| **Gradient Boosting Classifier (Best)** 🏆 | **83.84%** | **0.7752** | **0.8783** |
| Random Forest Classifier | 83.50% | 0.7705 | 0.8737 |
| Decision Tree Classifier | 82.83% | 0.7642 | 0.8553 |
| Logistic Regression | 81.37% | 0.7513 | 0.8649 |

---

## 🚀 যেকোনো পিসিতে রান ও টেস্ট করার উপায় (Execution Guide)

### পদ্ধতি ১: ১-ক্লিকে রান করা (Windows Users - Recommended)
১. প্রজেক্ট ফোল্ডারে থাকা **`run_app.bat`** ফাইলে ডাবল-ক্লিক করুন।
২. এটি স্বয়ংক্রিয়ভাবে প্রয়োজনীয় এনভায়রনমেন্ট তৈরি করবে, ডিপেন্ডেন্সি ইন্সটল করবে, মডেল ট্রেইন করবে এবং আপনার ডিফল্ট ব্রাউজারে **`http://127.0.0.1:5000/`** ওপেন করবে।

---

### পদ্ধতি ২: ম্যানুয়ালি টার্মিনালে রান করা

#### ১. প্রজেক্ট ফোল্ডারে টার্মিনাল ওপেন করুন এবং ভার্চুয়াল এনভায়রনমেন্ট সেটআপ করুন:
```bash
# Virtual Environment তৈরি করুন
py -m venv .venv

# Virtual Environment অ্যাক্টিভেট করুন (Windows)
.\.venv\Scripts\activate

# Linux / Mac এর ক্ষেত্রে:
# source .venv/bin/activate
```

#### ২. প্রয়োজনীয় ডিপেন্ডেন্সি ইন্সটল করুন:
```bash
pip install -r requirements.txt
```

#### ৩. মেশিন লার্নিং মডেল ট্রেইন করুন (Optional):
```bash
python src/train.py
```

#### ৪. Flask Web Application চালু করুন:
```bash
python app/app.py
```
এখন আপনার যেকোনো ব্রাউজার থেকে **`http://127.0.0.1:5000/`** অ্যাড্রেসে গিয়ে অ্যাপটি ব্যবহার করতে পারবেন!

---

## 🧪 ডেমো টেস্ট কেস (Sample Test Scenarios)

অ্যাপ্লিকেশনের নিখুঁত প্রেডিকশন পরীক্ষা করতে নিচের টেস্ট ইনপুটগুলো ইনপুট দিন:

### 🟢 টেস্ট কেস ১ (উচ্চ সম্ভাবনা):
- **Pclass**: 1st Class
- **Sex**: Female
- **Age**: 29
- **Fare**: $100
- **Embarked**: Cherbourg (C)
- 🎯 **ফলাফল**: `Survived ✅` (সম্ভাবনা: ~৯৯%)

### 🔴 টেস্ট কেস ২ (কম সম্ভাবনা):
- **Pclass**: 3rd Class
- **Sex**: Male
- **Age**: 30
- **Fare**: $8
- **Embarked**: Southampton (S)
- 🎯 **ফলাফল**: `Did Not Survive ❌` (সম্ভাবনা: ~১১%)

---

## 🐙 গিটহাবে পুশ করার নিয়ম (How to Push to GitHub)

যদি আপনি এই প্রজেক্টটি আপনার গিটহাব অ্যাকাউন্টে পুশ করতে চান, তবে টার্মিনালে পর্যায়ক্রমে নিচের কমান্ডগুলো রান করুন:

```bash
# ১. গিট রিপোজিটরি ইনিশিয়ালাইজ করুন
git init

# ২. মেইন ব্রাঞ্চ নির্বাচন করুন
git branch -M main

# ৩. সমস্ত ফাইল স্টেজিং অরিয়ায় যুক্ত করুন
git add .

# ৪. গিট কমিট করুন
git commit -m "Upgrade Titanic ML project to intermediate pipeline & Flask Web App"

# ৫. আপনার গিটহাব রিমোট রিপোজিটরি ইউআরএল যুক্ত করুন (YOUR_USERNAME ও YOUR_REPOSITORY জায়গায় নাম বসান)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git

# ৬. গিটহাবে পুশ করুন
git push -u origin main
```

---

## 👨‍💻 টেকনোলজি স্ট্যাক (Tech Stack)

- **Language**: Python 3.10+
- **Machine Learning**: Scikit-Learn, Pandas, NumPy, Joblib
- **Visualization**: Matplotlib, Seaborn
- **Web Framework**: Flask, HTML5, CSS3 (Glassmorphism), FontAwesome
- **Environment**: Virtualenv, Windows Batch Scripting

---

❤️ **প্রজেক্টটি ভালো লাগলে গিটহাবে একটি Star ⭐ দিন!**

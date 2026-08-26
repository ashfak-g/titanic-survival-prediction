import os
import sys
import pandas as pd
import numpy as np
import joblib

# Ensure UTF-8 output encoding for Windows terminal compatibility
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.feature_engineering import TitanicFeatureEngineer

def main():
    print("=" * 60)
    print(" Titanic Survival Prediction - Model Training Pipeline ")
    print("=" * 60)

    # 1. Load Data
    raw_train_path = os.path.join('data', 'raw', 'train.csv')
    raw_test_path = os.path.join('data', 'raw', 'test.csv')
    
    if not os.path.exists(raw_train_path):
        raise FileNotFoundError(f"Training file not found at {raw_train_path}")

    train_df = pd.read_csv(raw_train_path)
    test_df = pd.read_csv(raw_test_path)

    X_train = train_df.drop('Survived', axis=1)
    y_train = train_df['Survived']

    print(f"Loaded Raw Train Dataset: {X_train.shape[0]} rows, {X_train.shape[1]} columns")
    print(f"Loaded Raw Test Dataset:  {test_df.shape[0]} rows, {test_df.shape[1]} columns\n")

    # 2. Define Candidate Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    }

    # 3. Model Evaluation using 5-Fold Stratified Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print("Evaluating Candidate Models (5-Fold Stratified CV):")
    print("-" * 60)
    
    best_model_name = None
    best_score = 0.0

    for name, clf in models.items():
        pipeline = Pipeline([
            ('engineer', TitanicFeatureEngineer()),
            ('classifier', clf)
        ])
        
        cv_res = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=['accuracy', 'f1', 'roc_auc'])
        mean_acc = cv_res['test_accuracy'].mean()
        mean_f1 = cv_res['test_f1'].mean()
        mean_auc = cv_res['test_roc_auc'].mean()

        results[name] = {
            'Accuracy': mean_acc,
            'F1-Score': mean_f1,
            'ROC-AUC': mean_auc
        }
        
        print(f" -> {name:20s} | Accuracy: {mean_acc:.4f} | F1: {mean_f1:.4f} | AUC: {mean_auc:.4f}")
        
        if mean_acc > best_score:
            best_score = mean_acc
            best_model_name = name

    print("-" * 60)
    print(f"Best Model: {best_model_name} (Accuracy: {best_score:.4f})\n")

    # 4. Train Final Pipeline on Full Data
    best_clf = models[best_model_name]
    final_pipeline = Pipeline([
        ('engineer', TitanicFeatureEngineer()),
        ('classifier', best_clf)
    ])
    final_pipeline.fit(X_train, y_train)

    # 5. Save Artifacts
    os.makedirs('models', exist_ok=True)
    model_save_path = os.path.join('models', 'titanic_pipeline.pkl')
    joblib.dump(final_pipeline, model_save_path)
    print(f"Saved trained pipeline to: {model_save_path}")

    # 6. Generate Kaggle Test Predictions
    test_preds = final_pipeline.predict(test_df)
    submission_df = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': test_preds
    })
    
    os.makedirs(os.path.join('data', 'processed'), exist_ok=True)
    submission_path = os.path.join('data', 'processed', 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    print(f"Generated Kaggle Submission file: {submission_path}")

    print("\nTraining pipeline executed successfully!")

if __name__ == '__main__':
    main()

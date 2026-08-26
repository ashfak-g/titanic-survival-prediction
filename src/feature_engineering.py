import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Transformer for Titanic Dataset Feature Engineering.
    Extracts Titles, imputes missing values, engineers FamilySize and IsAlone,
    and maps categorical variables into numerical values.
    """
    def __init__(self):
        self.title_medians_ = {}
        self.pclass_fare_medians_ = {}
        self.embarked_mode_ = 'S'
        
    def fit(self, X, y=None):
        X_df = X.copy()
        
        # Extract Title
        if 'Name' in X_df.columns:
            titles = X_df['Name'].astype(str).str.extract(r' ([A-Za-z]+)\.', expand=False)
            titles = titles.replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
            titles = titles.replace('Mlle', 'Miss')
            titles = titles.replace('Ms', 'Miss')
            titles = titles.replace('Mme', 'Mrs')
            X_df['Title'] = titles
        else:
            X_df['Title'] = 'Mr'
            
        # Calculate medians for Age by Title
        self.title_medians_ = X_df.groupby('Title')['Age'].median().to_dict()
        self.global_age_median_ = float(X_df['Age'].median()) if not np.isnan(X_df['Age'].median()) else 28.0
        
        # Calculate medians for Fare by Pclass
        self.pclass_fare_medians_ = X_df.groupby('Pclass')['Fare'].median().to_dict()
        self.global_fare_median_ = float(X_df['Fare'].median()) if not np.isnan(X_df['Fare'].median()) else 14.45
        
        # Mode for Embarked
        if 'Embarked' in X_df.columns and not X_df['Embarked'].dropna().empty:
            self.embarked_mode_ = str(X_df['Embarked'].mode()[0])
            
        return self

    def transform(self, X):
        X_df = X.copy()
        
        # 1. Extract Title
        if 'Name' in X_df.columns:
            titles = X_df['Name'].astype(str).str.extract(r' ([A-Za-z]+)\.', expand=False)
            titles = titles.replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
            titles = titles.replace('Mlle', 'Miss')
            titles = titles.replace('Ms', 'Miss')
            titles = titles.replace('Mme', 'Mrs')
            X_df['Title'] = titles
        elif 'Title' not in X_df.columns:
            X_df['Title'] = 'Mr'
            
        # 2. Impute missing Age using Title medians
        if 'Age' in X_df.columns:
            def fill_age(row):
                if pd.isna(row['Age']):
                    return self.title_medians_.get(row.get('Title', 'Mr'), self.global_age_median_)
                return float(row['Age'])
            X_df['Age'] = X_df.apply(fill_age, axis=1)
        else:
            X_df['Age'] = self.global_age_median_
            
        # 3. Impute missing Fare using Pclass medians
        if 'Fare' in X_df.columns:
            def fill_fare(row):
                if pd.isna(row['Fare']):
                    return self.pclass_fare_medians_.get(row.get('Pclass', 3), self.global_fare_median_)
                return float(row['Fare'])
            X_df['Fare'] = X_df.apply(fill_fare, axis=1)
        else:
            X_df['Fare'] = self.global_fare_median_
            
        # 4. Impute Embarked
        if 'Embarked' in X_df.columns:
            X_df['Embarked'] = X_df['Embarked'].fillna(self.embarked_mode_)
        else:
            X_df['Embarked'] = self.embarked_mode_

        # 5. Map Sex: female=1, male=0
        sex_map = {'male': 0, 'female': 1, 'Male': 0, 'Female': 1, 0: 0, 1: 1}
        X_df['Sex'] = X_df['Sex'].astype(str).str.lower().map({'male': 0, 'female': 1}).fillna(0).astype(int)
            
        # 6. Map Embarked: S=0, C=1, Q=2
        emb_map = {'s': 0, 'c': 1, 'q': 2}
        X_df['Embarked'] = X_df['Embarked'].astype(str).str.lower().map(emb_map).fillna(0).astype(int)
            
        # 7. Map Title: Mr=0, Miss=1, Mrs=2, Master=3, Rare=4
        title_map = {'mr': 0, 'miss': 1, 'mrs': 2, 'master': 3, 'rare': 4}
        X_df['Title'] = X_df['Title'].astype(str).str.lower().map(title_map).fillna(0).astype(int)

        # 8. Feature Engineering: FamilySize & IsAlone
        sibsp = X_df['SibSp'].astype(int) if 'SibSp' in X_df.columns else 0
        parch = X_df['Parch'].astype(int) if 'Parch' in X_df.columns else 0
        X_df['FamilySize'] = sibsp + parch + 1
        X_df['IsAlone'] = (X_df['FamilySize'] == 1).astype(int)

        # 9. Select final numeric feature columns
        feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone']
        return X_df[feature_cols].astype(float)

def preprocess_single_input(pclass, sex, age, sibsp, parch, fare, embarked, title=None):
    """
    Helper function to format user inputs from web interface into a DataFrame.
    """
    pclass = int(pclass)
    sex_str = str(sex).strip().lower()
    age = float(age)
    sibsp = int(sibsp)
    parch = int(parch)
    fare = float(fare)
    embarked_str = str(embarked).strip().upper()

    if title is None or str(title).strip() == '':
        if sex_str == 'female':
            title = 'Mrs' if age > 18 else 'Miss'
        else:
            title = 'Master' if age < 14 else 'Mr'
            
    df = pd.DataFrame([{
        'Pclass': pclass,
        'Sex': sex_str,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked_str,
        'Title': title
    }])
    return df

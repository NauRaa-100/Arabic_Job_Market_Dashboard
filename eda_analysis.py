"""
سكريبت احترافى !
End-to-end pipeline for Arabic job-listings dataset
File: portfolio_ready.py
Purpose: Clean, normalize, feature-engineer, analyze, visualize and model
Input: CSV file (path hard-coded or passed as arg)
Output: cleaned CSV, EDA images, clustering labels, models, report (Excel + markdown)

How to use:
- Put your CSV at path or pass via argument
- Install requirements (requirements.txt provided in comments)
- Run: python portfolio_ready.py --input a.csv --output_dir outputs

This script is written to be copy-paste runnable and clear for a portfolio.
It handles Arabic normalization, salary parsing, clustering, TF-IDF, simple models
(LightGBM for salary regression; LogisticRegression for category classification)
and saves artifacts for inclusion in a portfolio.

Notes:
- Adjust `FONT_NAME` to a font installed on your machine to correctly render Arabic on plots.
- For heavy NLP models (AraBERT) add instructions; here we use TF-IDF + classical models to keep reproducible.

"""

import argparse
import os
import re
import json
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns

# Optional: LightGBM for regression (fast & common in industry)
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

warnings.filterwarnings('ignore')

# ------------------------- Config -------------------------
FONT_NAME = 'DejaVu Sans'  # change to 'Tahoma' or 'Arial' if available on your system
sns.set(style='darkgrid')
plt.rcParams['font.family'] = FONT_NAME

RANDOM_STATE = 42

# ------------------------- Helpers -------------------------
ARABIC_COLS = ['job_title','location','profession','description','job_category','sub_category']

RE_TASHKEEL = re.compile(r'[ًٌٍَُِّّْ]')

def normalize_ar(text):
    text = str(text) if pd.notnull(text) else ''
    text = text.strip()
    text = RE_TASHKEEL.sub('', text)
    # unify alef
    text = text.replace('أ','ا').replace('إ','ا').replace('آ','ا')
    text = text.replace('ى','ي').replace('ة','ه')
    # collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    return text


def extract_experience(text):
    text = str(text)
    nums = re.findall(r"(\d+)\s*(?:سنة|سنوات|عام|year|yrs|y)?", text)
    if nums:
        return int(nums[0])
    return np.nan


def extract_salary_range(text):
    text = str(text)
    # replace Arabic-Indic digits
    trans = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
    text = text.translate(trans)
    # extract numbers with optional thousands separators
    numbers = re.findall(r"\d{1,3}(?:[,\.]\d{3})*|\d+", text.replace(',', ''))
    numbers = [re.sub(r'[^0-9]', '', n) for n in numbers]
    numbers = [int(n) for n in numbers if n.isdigit()]
    if len(numbers) >= 2:
        return numbers[0], numbers[1]
    elif len(numbers) == 1:
        return numbers[0], numbers[0]
    else:
        return np.nan, np.nan

# ------------------------- Main pipeline -------------------------

def run_pipeline(input_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('Loading dataset...')
    df = pd.read_csv(input_path, encoding='utf-8')
    print('shape', df.shape)

    # Quick sanity: show columns
    print('columns:', df.columns.tolist())

    # Make a working copy
    df = df.copy()

    # Ensure text columns are strings
    for c in ARABIC_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str)

    # Fill simple missing for non-text numeric cols later
    # Memory optimization (example)
    for col in df.columns.difference(['job_title','description']):
        if pd.api.types.is_float_dtype(df[col].dtype):
            df[col] = df[col].astype('float32')
        elif pd.api.types.is_integer_dtype(df[col].dtype):
            df[col] = df[col].astype('int32')
        elif pd.api.types.is_object_dtype(df[col].dtype):
            df[col] = df[col].astype('category')

    # ----------------- Normalization -----------------
    print('Normalizing Arabic text...')
    for col in ARABIC_COLS:
        if col in df.columns:
            df[col + '_norm'] = df[col].apply(normalize_ar)

    # ----------------- Salary parsing -----------------
    # We expect columns: 'salary' (text range), 'salary_local' (numeric), 'salary_usd' (numeric)
    if 'salary' in df.columns:
        df['salary'] = df['salary'].astype(str)
        ranges = df['salary'].apply(extract_salary_range)
        df['salary_min'] = ranges.apply(lambda x: x[0])
        df['salary_max'] = ranges.apply(lambda x: x[1])
        df['salary_avg'] = (df['salary_min'] + df['salary_max']) / 2

    # If salary_local or salary_usd exist and are numeric but have NaNs, keep them as-is
    # Create a chosen salary column for analysis: prefer salary_avg -> salary_local -> salary_usd
    def pick_salary_row(r):
        if pd.notnull(r.get('salary_avg')):
            return r['salary_avg']
        if 'salary_local' in r and pd.notnull(r['salary_local']):
            return r['salary_local']
        if 'salary_usd' in r and pd.notnull(r['salary_usd']):
            return r['salary_usd']
        return np.nan

    df['salary_chosen'] = df.apply(pick_salary_row, axis=1)

    # ----------------- Experience extraction -----------------
    if 'description' in df.columns:
        df['experience_years'] = df['description'].apply(extract_experience)

    # ----------------- Encoding gender -----------------
    if 'gender' in df.columns:
        df['gender'] = df['gender'].astype(str).str.lower()
        df['gender'] = df['gender'].replace({'f': 'female', 'm': 'male'})

    # ----------------- Clustering job_title -----------------
    print('TF-IDF & clustering job titles...')
    if 'job_title_norm' in df.columns:
        vec = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
        X = vec.fit_transform(df['job_title_norm'].fillna(''))
        n_clusters = 20
        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
        df['job_title_cluster'] = kmeans.fit_predict(X)
        # Save cluster terms sample
        terms = vec.get_feature_names_out()
        cluster_centers = kmeans.cluster_centers_ if hasattr(kmeans, 'cluster_centers_') else None
        # Save small mapping of cluster -> example titles
        cluster_examples = df.groupby('job_title_cluster')['job_title_norm'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]).to_dict()
        pd.Series(cluster_examples).to_csv(output_dir / 'cluster_examples.csv')

    # ----------------- Basic EDA & Visuals -----------------
    print('Running EDA and saving visuals...')

    # Jobs per location (top 20)
    if 'location_norm' in df.columns:
        jobs_loc = df.groupby('location_norm')['job_title'].count().sort_values(ascending=False).head(20)
        plt.figure(figsize=(10,6))
        sns.barplot(x=jobs_loc.values, y=jobs_loc.index)
        plt.title('Top 20 Locations by Number of Jobs')
        plt.tight_layout()
        plt.savefig(output_dir / 'jobs_by_location.png')
        plt.close()

    # Salary distributions
    if 'salary_chosen' in df.columns:
        plt.figure(figsize=(8,6))
        sns.histplot(df['salary_chosen'].dropna(), kde=True)
        plt.title('Salary Distribution (chosen)')
        plt.tight_layout()
        plt.savefig(output_dir / 'salary_distribution.png')
        plt.close()

    # Average salary by category
    if 'job_category_norm' in df.columns and 'salary_chosen' in df.columns:
        cat_salary = df.groupby('job_category_norm')['salary_chosen'].mean().dropna().sort_values(ascending=False).head(30)
        plt.figure(figsize=(12,8))
        sns.barplot(x=cat_salary.values, y=cat_salary.index)
        plt.title('Average Salary by Job Category (top 30)')
        plt.tight_layout()
        plt.savefig(output_dir / 'avg_salary_by_category.png')
        plt.close()

    # Gender distribution
    if 'gender' in df.columns:
        g = df['gender'].value_counts(normalize=True).mul(100)
        g.plot(kind='bar')
        plt.title('Gender distribution (%)')
        plt.tight_layout()
        plt.savefig(output_dir / 'gender_distribution.png')
        plt.close()

    # Save cleaned dataframe sample and full cleaned CSV
    df.to_csv(output_dir / 'cleaned_full.csv', index=False, encoding='utf-8')
    df.head(200).to_csv(output_dir / 'cleaned_sample_200.csv', index=False, encoding='utf-8')

    # ----------------- Modeling examples -----------------
    reports = {}

    # 1) Salary regression (predict salary_chosen) - if LightGBM available use it
    if LGB_AVAILABLE and 'salary_chosen' in df.columns and df['salary_chosen'].notna().sum() > 200:
        print('Training LightGBM regression for salary prediction...')
        fe_cols = []
        # simple features: experience_years, gender (encoded), job_category_norm (encoded)
        if 'experience_years' in df.columns:
            fe_cols.append('experience_years')
        if 'gender' in df.columns:
            df['gender_le'] = LabelEncoder().fit_transform(df['gender'].astype(str))
            fe_cols.append('gender_le')
        if 'job_category_norm' in df.columns:
            df['job_cat_le'] = LabelEncoder().fit_transform(df['job_category_norm'].astype(str))
            fe_cols.append('job_cat_le')

        # train/test
        sub = df[fe_cols + ['salary_chosen']].dropna()
        X = sub[fe_cols]
        y = sub['salary_chosen']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
        lgb_train = lgb.Dataset(X_train, y_train)
        params = {'objective': 'regression','metric':'rmse','verbosity':-1,'seed':RANDOM_STATE}
        gbm = lgb.train(params, lgb_train, num_boost_round=200)
        y_pred = gbm.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        reports['salary_regression'] = {'rmse': float(rmse), 'r2': float(r2)}
        # save model
        try:
            import joblib
            joblib.dump(gbm, output_dir / 'lgb_salary_model.pkl')
        except Exception:
            pass

    # 2) Classification example: predict job_category_norm from job_title_norm (simple)
    if 'job_category_norm' in df.columns:
        print('Training simple classifier for job_category from title (TF-IDF + LogisticRegression)...')
        df_clf = df[['job_title_norm','job_category_norm']].dropna()
        if len(df_clf) > 200:
            vec2 = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
            X = vec2.fit_transform(df_clf['job_title_norm'])
            le = LabelEncoder()
            y = le.fit_transform(df_clf['job_category_norm'])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            reports['classification'] = classification_report(y_test, y_pred, output_dict=True)
            # save classifier artifacts
            try:
                import joblib
                joblib.dump(clf, output_dir / 'clf_jobcat.pkl')
                joblib.dump(vec2, output_dir / 'tfidf_jobtitle.pkl')
                joblib.dump(le, output_dir / 'labelenc_jobcat.pkl')
            except Exception:
                pass

    # Save reports and a small README
    with open(output_dir / 'pipeline_report.json', 'w', encoding='utf-8') as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)

    readme = {
        'notes': 'This folder contains cleaned data, visualizations, model artifacts and a report. Use these assets in your portfolio.',
        'files': [str(p.name) for p in output_dir.iterdir()]
    }
    with open(output_dir / 'README_pipeline.json', 'w', encoding='utf-8') as f:
        json.dump(readme, f, ensure_ascii=False, indent=2)

    print('Pipeline finished. Outputs saved to', output_dir)


# ------------------------- CLI -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='input CSV path')
    parser.add_argument('--output_dir', type=str, default='outputs', help='folder to save outputs')
    args = parser.parse_args()
    run_pipeline(args.input, args.output_dir)

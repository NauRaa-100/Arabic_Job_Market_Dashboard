

# README —> Arabic Job Market Dashboard

##  Overview

The **Arabic Job Market Dashboard** is a complete data science project that analyzes job market trends across the Arab region.
It includes:

* Full data cleaning & preprocessing
* Exploratory data analysis (EDA)
* Machine learning models for job category prediction and salary estimation
* Interactive dashboard built with **Gradio + Plotly**
* Ready-to-use models and visualizations

This project simulates real company-level work, combining data engineering, ML pipelines, and UI deployment.

---

## Project Structure

```
project/
│
├── cleaned_full.csv                # Cleaned dataset
│
├── analysis.ipynb                  # EDA & data exploration│
├── tfidf_jobtitle.pkl              # TF-IDF vectorizer
├── clf_jobcat.pkl                  # Job category classifier
├── labelenc_jobcat.pkl             # Label encoder
├── lgb_salary_model.pkl            # LightGBM salary model
│
├── dashboard.py                    # Gradio dashboard code
│
├── requirements.txt
└── README.md
```

---

##  1. Data Cleaning & Preprocessing

The dataset was cleaned and normalized with the following steps:

###  Remove duplicates

###  Handle missing values

* Salary: replaced using statistical imputations
* Job titles & categories: normalized and cleaned
* Locations: unified and corrected

###  Standardized columns

* `job_title_norm`
* `job_category_norm`
* `location_norm`
* Encoded gender (0/1)
* Converted experience into numerical format

### Extract job categories

Applied TF-IDF + Logistic Regression to classify job titles into normalized categories.

---

##  2. Exploratory Data Analysis (EDA)

The EDA explored multiple market patterns:

###  Top Hiring Locations

Frequency distribution of the most common job locations.

###  Salary Distribution

Histogram showing salary ranges across the region.

### Job Category Distribution

Understanding which fields dominate the job market.

### Experience vs Salary

Clear positive correlation between years of experience and expected salary.

> *(You may insert images here if needed)*

---

##  3. Machine Learning Models

### **A) Job Category Prediction**

**Model:** Logistic Regression
**Features:** TF-IDF representation of `job_title_norm`
**Output:**

* Predicted job category
* Top 5 most similar categories (suggestions)

---

### **B) Salary Prediction**

**Model:** LightGBM Regressor
**Features used:**

* Experience
* Gender
* Job category (encoded)

**Output:**

* Numerical salary prediction, rounded to 2 decimal places

---

## 4. Interactive Dashboard (Gradio)

The dashboard includes 4 main sections:

###  **Charts**

Interactive Plotly charts allowing:

* Zoom
* Pan
* Hover labels

Charts available:

* Top 10 Locations
* Salary Distribution

---

###  **Suggest Job Categories**

* User enters/selects a job title
* Model returns the most likely category + top 5 alternatives

---

###  **Predict Job Category**

* Predicts the category from job title

---

###  **Predict Salary**

Inputs:

* Job title
* Experience
* Gender
* Job category

Outputs:

* Estimated salary

---


##  5. How to Run Locally

###  Install dependencies:

```
pip install -r requirements.txt
```

###  Run the dashboard:

```
python dashboard.py
```

Once it launches, open:

```
http://127.0.0.1:7860
```

---

##  6. Requirements

```
numpy
pandas
scikit-learn
seaborn
matplotlib
plotly
gradio
joblib
re
```

( Note: `json` is part of Python and should *not* be installed.)

---

##  7. Future Improvements

* Add BERT-based job title classification
* Add a model predicting job location
* Enhance feature engineering for salaries
* Add Power BI or Tableau dashboard

---

##  Conclusion

This project provides an end-to-end data science pipeline, including:

* Data cleaning
* EDA
* ML models
* Interactive dashboard

It simulates real-world machine learning workflows used in companies and provides practical insights into the Arab job market.

import gradio as gr
import pandas as pd
import joblib
import plotly.express as px

# ---------------------------------------------------
# Load cleaned data
# ---------------------------------------------------
df = pd.read_csv("cleaned_full.csv")

# ---------------------------------------------------
# Load models if available
# ---------------------------------------------------
try:
    clf = joblib.load("clf_jobcat.pkl")
    tfidf = joblib.load("tfidf_jobtitle.pkl")
    le = joblib.load("labelenc_jobcat.pkl")
except:
    clf = None

try:
    gbm = joblib.load("lgb_salary_model.pkl")
except:
    gbm = None

# ---------------------------------------------------
# Extract unique mappings
# ---------------------------------------------------
job_categories = sorted([x for x in df["job_category_norm"].dropna().unique()])
locations = sorted([x for x in df["location_norm"].dropna().unique()])
job_titles = sorted(df["job_title_norm"].dropna().unique())

# ---------------------------------------------------
# Prediction functions
# ---------------------------------------------------

def suggest_categories(title):
    if clf is None:
        return ["Model not available"]
    X = tfidf.transform([title])
    pred = clf.predict(X)[0]
    main = le.inverse_transform([pred])[0]
    return [main] + job_categories[:5]

def predict_category(title):
    if clf is None:
        return "Model not available"
    X = tfidf.transform([title])
    pred = clf.predict(X)[0]
    return le.inverse_transform([pred])[0]

def predict_salary(job_title, experience, gender, category):
    if gbm is None:
        return "Salary model not available"

    gender_map = {"male": 1, "female": 0}
    gender_val = gender_map.get(gender.lower(), 0)

    cat_list = df["job_category_norm"].astype(str).unique().tolist()
    try:
        cat_val = cat_list.index(category)
    except:
        cat_val = 0

    row = pd.DataFrame([{
        "experience_years": float(experience),
        "gender_le": gender_val,
        "job_cat_le": cat_val
    }])

    pred = gbm.predict(row)[0]
    return round(pred, 2)  

# ---------------------------------------------------
# Charts using Plotly for interactive zoom
# ---------------------------------------------------

def chart_locations():
    top10 = df["location_norm"].value_counts().head(10).reset_index()
    top10.columns = ["Location", "Count"]
    fig = px.bar(top10, x="Count", y="Location", orientation='h',
                 title="Top 10 Locations", text="Count")
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig

def chart_salary():
    fig = px.histogram(df, x="salary_chosen", nbins=40, title="Salary Distribution")
    return fig

# ---------------------------------------------------
# Gradio App
# ---------------------------------------------------

with gr.Blocks(title="Arabic Jobs Market Dashboard") as demo:

    gr.Markdown("# 🌍 Arabic Jobs Market Dashboard")

    with gr.Tab("📊 Charts"):
        gr.Plot(chart_locations)
        gr.Plot(chart_salary)

    with gr.Tab("🧠 Suggest Job Categories"):
        jt = gr.Dropdown(label="اكتب عنوان الوظيفة", choices=job_titles, interactive=True)
        out_sg = gr.JSON(label="اقتراحات مبنية على الموديل ")
        gr.Button("اقترح").click(suggest_categories, jt, out_sg)

    with gr.Tab("🗂 Predict Job Category"):
        jt2 = gr.Dropdown(label="Job Title", choices=job_titles, interactive=True)
        out2 = gr.Textbox(label="Predicted Category")
        gr.Button("Predict").click(predict_category, jt2, out2)

    with gr.Tab("💰 Predict Salary"):
        job_title = gr.Dropdown(label="Job Title", choices=job_titles, interactive=True)
        exp = gr.Number(label="Experience", value=1)
        gender = gr.Radio(["male", "female"], label="Gender")
        category = gr.Dropdown(choices=job_categories, label="Job Category")
        out_sal = gr.Number(label="Predicted Salary")
        gr.Button("Predict").click(
            predict_salary,
            [job_title, exp, gender, category],
            out_sal
        )

demo.launch()

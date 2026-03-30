# 🎯 Lead Scoring Project

A machine learning project that predicts which leads are most likely to convert into paying customers — helping sales teams focus their efforts and increase conversion rates.

---

## 📌 Problem Statement

In sales and marketing, not all leads are equal. Companies waste time and resources chasing leads that will never convert. This project builds a **Lead Scoring System** that assigns each lead a conversion probability, so sales teams can prioritize high-potential leads.

---

## 🚀 Demo

> Upload your `Lead Scoring.csv` file into the Streamlit app and get instant predictions!

**App Pages:**
| Page | Description |
|------|-------------|
| 📊 Home | Overview and key stats |
| 📁 Data Upload | Upload CSV and explore data |
| 🔍 EDA | Visualizations and correlation analysis |
| 🤖 Model Training | Train 3 ML models with configurable settings |
| 📈 Results | Compare model performance, confusion matrix, ROC curve |
| 🎯 Predictions | Enter lead info and get real-time conversion prediction |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.x |
| Web App | Streamlit |
| ML Models | Scikit-learn (Logistic Regression, Random Forest, Gradient Boosting) |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Model Persistence | Joblib |

---

## 📁 Project Structure

```
Lead Scoring Project/
│
├── app.py                        # Streamlit web application
├── pipeline.py                   # Full ML pipeline (standalone script)
├── Lead Scoring.csv              # Raw dataset
├── requirements.txt              # Python dependencies
├── .gitignore                    # Files excluded from Git
│
└── lead_scoring_output/
    ├── models/
    │   └── best_lead_scoring_model.joblib   # Saved best model
    ├── plots/
    │   ├── target_distribution.png
    │   ├── lead_origin_vs_converted.png
    │   ├── time_spent_vs_converted.png
    │   ├── correlation_heatmap.png
    │   ├── feature_distributions.png
    │   ├── model_comparison.png
    │   ├── confusion_matrix.png
    │   └── roc_curve.png
    ├── cleaned_lead_scoring.csv
    └── model_comparison.csv
```

---

## ⚙️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/lead-scoring-project.git
cd lead-scoring-project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

### 4. (Optional) Run the Full Pipeline Script
```bash
python pipeline.py
```

---

## 🤖 ML Pipeline

1. **Data Loading** — Reads the raw CSV dataset
2. **Preprocessing** — Handles missing values, drops high-null columns, fills with median/mode
3. **Feature Engineering** — Groups rare categories in `Lead Source` and `Lead Origin`
4. **EDA** — Visualizes target distribution, correlations, and feature patterns
5. **Model Training** — Trains 3 models with a `ColumnTransformer` + `Pipeline`
6. **Evaluation** — Accuracy, Precision, Recall, F1-Score, ROC-AUC
7. **Best Model Saved** — Exports the best model as `.joblib`

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 92.97% | 92.42% | 89.04% | 90.70% | 97.55% |
| Random Forest | 93.40% | 92.14% | 90.59% | 91.36% | 97.58% |
| Gradient Boosting | 94.10% | 93.76% | 90.73% | 92.22% | 97.94% |

> Run the pipeline to get exact metrics on your data.

---

## 💡 Key Insights

- Leads who spend **more time on the website** are significantly more likely to convert
- **Lead Origin** (e.g., Landing Page Submission) is a strong predictor of conversion
- Gradient Boosting consistently outperforms other models on this dataset

---

## 📬 Contact

Feel free to connect or reach out if you have questions about the project!

- GitHub: [your-username](https://github.com/your-username)
- LinkedIn: [your-linkedin]([https://linkedin.com/in/your-profile](https://www.linkedin.com/in/youssef-gendia-562994349))

---

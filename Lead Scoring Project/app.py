"""
================================================================================
LEAD SCORING STREAMLIT APP
================================================================================
A complete Streamlit application for lead scoring data analysis and ML modeling
Run with: streamlit run app.py
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    roc_auc_score, roc_curve, precision_score, recall_score, f1_score
)
import joblib

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Lead Scoring Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df = None
    st.session_state.df_clean = None
    st.session_state.models_trained = False
    st.session_state.results = None
    st.session_state.best_model = None
    st.session_state.best_model_name = None
    st.session_state.X_test = None
    st.session_state.y_test = None

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.title("🎯 Lead Scoring Platform")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Home", "📁 Data Upload", "🔍 EDA", "🤖 Model Training", "📈 Results", "🎯 Predictions"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This app performs complete lead scoring analysis including data preprocessing, "
    "exploratory data analysis, and machine learning model training."
)

# ============================================================================
# PAGE 1: HOME
# ============================================================================
if page == "📊 Home":
    st.title("🎯 Lead Scoring & Prediction Platform")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Welcome to Lead Scoring Platform!
        
        This comprehensive application helps you:
        
        ✅ **Upload & Explore** your lead data
        
        ✅ **Analyze** patterns and relationships
        
        ✅ **Train** multiple ML models
        
        ✅ **Compare** model performance
        
        ✅ **Make Predictions** on new leads
        
        ✅ **Export** results and models
        """)
    
    with col2:
        st.markdown("""
        ### Features
        
        📊 **Data Exploration**
        - Missing value analysis
        - Statistical summaries
        - Data quality checks
        
        📈 **Visualizations**
        - Target distribution
        - Feature correlations
        - Conversion analysis
        
        🤖 **ML Models**
        - Logistic Regression
        - Random Forest
        - Gradient Boosting
        
        📊 **Performance Metrics**
        - Accuracy, Precision, Recall
        - F1-Score, ROC-AUC
        - Confusion Matrix
        """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Models Available", "3", "LR, RF, GB")
    
    with col2:
        st.metric("Visualizations", "5+", "EDA Charts")
    
    with col3:
        st.metric("Metrics Tracked", "6", "Accuracy, Precision, etc.")

# ============================================================================
# PAGE 2: DATA UPLOAD
# ============================================================================
elif page == "📁 Data Upload":
    st.title("📁 Data Upload & Exploration")
    
    st.markdown("### Step 1: Upload Your CSV File")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.data_loaded = True
            
            st.success("✓ Data loaded successfully!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", f"{df.shape[0]:,}")
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            st.markdown("---")
            st.markdown("### Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown("---")
            st.markdown("### Data Info")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Data Types:**")
                st.dataframe(pd.DataFrame(df.dtypes, columns=['Type']), use_container_width=True)
            
            with col2:
                st.markdown("**Missing Values:**")
                missing_data = pd.DataFrame({
                    'Column': df.columns,
                    'Missing_Count': df.isnull().sum(),
                    'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
                })
                missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
                
                if len(missing_data) > 0:
                    st.dataframe(missing_data, use_container_width=True)
                else:
                    st.info("No missing values found!")
            
            st.markdown("---")
            st.markdown("### Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)
            
            # Target variable analysis
            if 'Converted' in df.columns:
                st.markdown("---")
                st.markdown("### Target Variable Distribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    target_counts = df['Converted'].value_counts()
                    st.dataframe(target_counts, use_container_width=True)
                
                with col2:
                    target_pct = df['Converted'].value_counts(normalize=True) * 100
                    st.dataframe(target_pct, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    else:
        st.info("👆 Upload a CSV file to get started")

# ============================================================================
# PAGE 3: EDA
# ============================================================================
elif page == "🔍 EDA":
    st.title("🔍 Exploratory Data Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("Please upload data first in the Data Upload section")
    else:
        df = st.session_state.df
        
        # Preprocessing
        df_clean = df.copy()
        df_clean = df_clean.replace('Select', np.nan)
        
        missing_percent = (df_clean.isnull().sum() / len(df_clean)) * 100
        cols_to_drop = missing_percent[missing_percent > 40].index.tolist()
        df_clean = df_clean.drop(columns=cols_to_drop, errors='ignore')
        
        unique_counts = df_clean.nunique()
        cols_to_drop_single = unique_counts[unique_counts <= 1].index.tolist()
        df_clean = df_clean.drop(columns=cols_to_drop_single, errors='ignore')
        
        id_cols = ['Prospect ID', 'Lead Number', 'ID']
        id_cols_present = [col for col in id_cols if col in df_clean.columns]
        df_clean = df_clean.drop(columns=id_cols_present, errors='ignore')
        
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df_clean[col] = df_clean[col].fillna('Unknown')
        
        numerical_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        st.session_state.df_clean = df_clean
        
        st.success("✓ Data preprocessed successfully!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Rows", df.shape[0])
        with col2:
            st.metric("Cleaned Rows", df_clean.shape[0])
        with col3:
            st.metric("Columns Removed", df.shape[1] - df_clean.shape[1])
        
        st.markdown("---")
        
        # Visualizations
        st.markdown("### Visualizations")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Target Distribution", "Lead Origin", "Time Spent", "Correlations", "Feature Dist"]
        )
        
        # Tab 1: Target Distribution
        with tab1:
            if 'Converted' in df_clean.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                target_counts = df_clean['Converted'].value_counts()
                colors = ['#FF6B6B', '#4ECDC4']
                ax.bar(target_counts.index, target_counts.values, color=colors, edgecolor='black', linewidth=1.5)
                ax.set_title('Distribution of Target Variable (Converted)', fontsize=14, fontweight='bold')
                ax.set_xlabel('Converted')
                ax.set_ylabel('Count')
                ax.set_xticklabels(['Not Converted', 'Converted'])
                for i, v in enumerate(target_counts.values):
                    ax.text(i, v + 50, str(v), ha='center', fontweight='bold')
                st.pyplot(fig)
        
        # Tab 2: Lead Origin
        with tab2:
            if 'Lead Origin' in df_clean.columns and 'Converted' in df_clean.columns:
                fig, ax = plt.subplots(figsize=(14, 7))
                sns.countplot(data=df_clean, x='Lead Origin', hue='Converted', palette=['#FF6B6B', '#4ECDC4'], ax=ax)
                ax.set_title('Lead Origin vs Conversion Status', fontsize=14, fontweight='bold')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.legend(title='Converted', labels=['No', 'Yes'])
                st.pyplot(fig)
        
        # Tab 3: Time Spent
        with tab3:
            if 'Total Time Spent on Website' in df_clean.columns and 'Converted' in df_clean.columns:
                fig, ax = plt.subplots(figsize=(12, 7))
                sns.boxplot(data=df_clean, x='Converted', y='Total Time Spent on Website', 
                           palette=['#FF6B6B', '#4ECDC4'], ax=ax)
                ax.set_title('Time Spent on Website vs Conversion', fontsize=14, fontweight='bold')
                ax.set_xticklabels(['Not Converted', 'Converted'])
                st.pyplot(fig)
        
        # Tab 4: Correlations
        with tab4:
            numerical_cols_all = df_clean.select_dtypes(include=['float64', 'int64']).columns
            if len(numerical_cols_all) > 1:
                fig, ax = plt.subplots(figsize=(12, 10))
                correlation_matrix = df_clean[numerical_cols_all].corr()
                sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                           center=0, square=True, linewidths=1, ax=ax)
                ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
                st.pyplot(fig)
        
        # Tab 5: Feature Distributions
        with tab5:
            numerical_cols_all = df_clean.select_dtypes(include=['float64', 'int64']).columns
            if len(numerical_cols_all) > 0:
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                axes = axes.flatten()
                
                for idx, col in enumerate(numerical_cols_all[:4]):
                    axes[idx].hist(df_clean[col], bins=30, color='#4ECDC4', edgecolor='black', alpha=0.7)
                    axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
                    axes[idx].set_xlabel(col)
                    axes[idx].set_ylabel('Frequency')
                
                st.pyplot(fig)

# ============================================================================
# PAGE 4: MODEL TRAINING
# ============================================================================
elif page == "🤖 Model Training":
    st.title("🤖 Model Training")
    
    if not st.session_state.data_loaded:
        st.warning("Please upload data first")
    else:
        if st.session_state.df_clean is None:
            st.warning("Please run EDA first to preprocess data")
        else:
            df_clean = st.session_state.df_clean
            
            st.markdown("### Training Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
            
            with col2:
                random_state = st.number_input("Random State", 0, 1000, 42)
            
            if st.button("🚀 Train Models", key="train_button", use_container_width=True):
                with st.spinner("Training models... This may take a minute..."):
                    try:
                        # Prepare data
                        X = df_clean.drop(columns=['Converted'])
                        y = df_clean['Converted']
                        
                        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
                        numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
                        
                        # Preprocessing pipeline
                        preprocessor = ColumnTransformer(
                            transformers=[
                                ('num', StandardScaler(), numerical_cols),
                                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
                            ],
                            remainder='passthrough'
                        )
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=random_state, stratify=y
                        )
                        
                        st.session_state.X_test = X_test
                        st.session_state.y_test = y_test
                        
                        # Define models
                        models = {
                            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
                            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
                            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state)
                        }
                        
                        results = {}
                        best_model = None
                        best_score = 0
                        best_model_name = None
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for idx, (model_name, model) in enumerate(models.items()):
                            status_text.text(f"Training {model_name}...")
                            
                            pipeline = Pipeline(steps=[
                                ('preprocessor', preprocessor),
                                ('model', model)
                            ])
                            
                            pipeline.fit(X_train, y_train)
                            
                            y_pred = pipeline.predict(X_test)
                            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                            
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred)
                            recall = recall_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred)
                            roc_auc = roc_auc_score(y_test, y_pred_proba)
                            
                            results[model_name] = {
                                'accuracy': accuracy,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1,
                                'roc_auc': roc_auc,
                                'model': pipeline,
                                'y_pred': y_pred,
                                'y_pred_proba': y_pred_proba
                            }
                            
                            if accuracy > best_score:
                                best_score = accuracy
                                best_model = pipeline
                                best_model_name = model_name
                            
                            progress_bar.progress((idx + 1) / len(models))
                        
                        st.session_state.models_trained = True
                        st.session_state.results = results
                        st.session_state.best_model = best_model
                        st.session_state.best_model_name = best_model_name
                        
                        st.success("✓ Models trained successfully!")
                        
                    except Exception as e:
                        st.error(f"Error during training: {str(e)}")

# ============================================================================
# PAGE 5: RESULTS
# ============================================================================
elif page == "📈 Results":
    st.title("📈 Model Results & Comparison")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first")
    else:
        results = st.session_state.results
        best_model_name = st.session_state.best_model_name
        y_test = st.session_state.y_test
        
        # Model Comparison
        st.markdown("### Model Performance Comparison")
        
        comparison_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [results[m]['accuracy'] for m in results.keys()],
            'Precision': [results[m]['precision'] for m in results.keys()],
            'Recall': [results[m]['recall'] for m in results.keys()],
            'F1-Score': [results[m]['f1'] for m in results.keys()],
            'ROC-AUC': [results[m]['roc_auc'] for m in results.keys()]
        })
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualize comparison
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        for idx, metric in enumerate(metrics):
            axes[idx].bar(comparison_df['Model'], comparison_df[metric], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            axes[idx].set_title(metric, fontweight='bold')
            axes[idx].set_ylim([0, 1])
            axes[idx].tick_params(axis='x', rotation=45)
            for i, v in enumerate(comparison_df[metric]):
                axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold', fontsize=8)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        st.markdown(f"### Best Model: {best_model_name}")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{results[best_model_name]['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{results[best_model_name]['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{results[best_model_name]['recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{results[best_model_name]['f1']:.4f}")
        with col5:
            st.metric("ROC-AUC", f"{results[best_model_name]['roc_auc']:.4f}")
        
        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["Classification Report", "Confusion Matrix", "ROC Curve"])
        
        with tab1:
            y_pred = results[best_model_name]['y_pred']
            report = classification_report(y_test, y_pred, target_names=['Not Converted', 'Converted'], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
        
        with tab2:
            y_pred = results[best_model_name]['y_pred']
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                       xticklabels=['Not Converted', 'Converted'],
                       yticklabels=['Not Converted', 'Converted'], ax=ax)
            ax.set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            st.pyplot(fig)
        
        with tab3:
            y_pred_proba = results[best_model_name]['y_pred_proba']
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = results[best_model_name]['roc_auc']
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, color='#4ECDC4', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
            ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve - {best_model_name}', fontsize=14, fontweight='bold')
            ax.legend(loc="lower right")
            ax.grid(alpha=0.3)
            st.pyplot(fig)

# ============================================================================
# PAGE 6: PREDICTIONS
# ============================================================================
elif page == "🎯 Predictions":
    st.title("🎯 Make Predictions")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first")
    else:
        st.markdown("### Enter Lead Information")
        
        best_model = st.session_state.best_model
        df_clean = st.session_state.df_clean
        
        # Get unique values for categorical columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        
        input_data = {}
        
        col1, col2 = st.columns(2)
        
        for idx, col in enumerate(df_clean.columns):
            if col == 'Converted':
                continue
            
            target_col = col1 if idx % 2 == 0 else col2
            
            if col in categorical_cols:
                unique_vals = df_clean[col].unique().tolist()
                input_data[col] = target_col.selectbox(f"{col}", unique_vals)
            else:
                min_val = df_clean[col].min()
                max_val = df_clean[col].max()
                input_data[col] = target_col.slider(f"{col}", float(min_val), float(max_val), float(df_clean[col].mean()))
        
        if st.button("🔮 Predict", use_container_width=True):
            try:
                # Create DataFrame from input
                new_lead = pd.DataFrame([input_data])
                
                # Make prediction
                prediction = best_model.predict(new_lead)[0]
                probability = best_model.predict_proba(new_lead)[0]
                
                st.markdown("---")
                st.markdown("### Prediction Result")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction == 1:
                        st.success("✓ CONVERTED")
                    else:
                        st.error("✗ NOT CONVERTED")
                
                with col2:
                    st.metric("Conversion Probability", f"{probability[1]:.2%}")
                
                with col3:
                    st.metric("Confidence", f"{max(probability):.2%}")
                
                st.markdown("---")
                
                # Probability breakdown
                fig, ax = plt.subplots(figsize=(10, 6))
                categories = ['Not Converted', 'Converted']
                colors = ['#FF6B6B', '#4ECDC4']
                ax.bar(categories, probability, color=colors, edgecolor='black', linewidth=1.5)
                ax.set_title('Conversion Probability Breakdown', fontsize=14, fontweight='bold')
                ax.set_ylabel('Probability')
                ax.set_ylim([0, 1])
                for i, v in enumerate(probability):
                    ax.text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
                st.pyplot(fig)
            
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    <p>Lead Scoring Platform | Built with Streamlit | 2026</p>
</div>
""", unsafe_allow_html=True)

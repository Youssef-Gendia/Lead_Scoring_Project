# ============================================================================
# LEAD SCORING _Full Pipeline 
# ============================================================================

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
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# CONFIGURATION
# ============================================================================
os.makedirs('lead_scoring_output', exist_ok=True)
os.makedirs('lead_scoring_output/plots', exist_ok=True)
os.makedirs('lead_scoring_output/models', exist_ok=True)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n" + "="*80)
print("1. LOADING DATA")
print("="*80)

# UPDATE THIS PATH TO YOUR CSV FILE
df = pd.read_csv(r'C:\Users\mosta\OneDrive\Desktop\Data Science Projects\Lead Scoring Project\Lead Scoring.csv')

print(f"✓ Data loaded successfully!")
print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# ============================================================================
# 2. DATA EXPLORATION
# ============================================================================
print("\n" + "="*80)
print("2. DATA EXPLORATION")
print("="*80)

print(f"\nData Types:\n{df.dtypes}\n")

missing_data = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
})
missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
if len(missing_data) > 0:
    print("Missing Values:")
    print(missing_data)
else:
    print("No missing values found")

select_counts = (df == 'Select').sum()
select_counts = select_counts[select_counts > 0]
if len(select_counts) > 0:
    print(f"\n'Select' Placeholder Values:")
    print(select_counts)

if 'Converted' in df.columns:
    print(f"\nTarget Variable Distribution:")
    print(df['Converted'].value_counts())
    print(f"\nTarget Percentage:")
    print(df['Converted'].value_counts(normalize=True) * 100)

# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================
print("\n" + "="*80)
print("3. DATA PREPROCESSING")
print("="*80)

df_clean = df.copy()

# Replace 'Select' with NaN
df_clean = df_clean.replace('Select', np.nan)
print("\n✓ Replaced 'Select' with NaN")

# Drop columns with >40% missing values
missing_percent = (df_clean.isnull().sum() / len(df_clean)) * 100
cols_to_drop = missing_percent[missing_percent > 40].index.tolist()
if cols_to_drop:
    print(f"✓ Dropped columns with >40% missing: {cols_to_drop}")
    df_clean = df_clean.drop(columns=cols_to_drop)

# Drop single-value columns
unique_counts = df_clean.nunique()
cols_to_drop_single = unique_counts[unique_counts <= 1].index.tolist()
if cols_to_drop_single:
    print(f"✓ Dropped single-value columns: {cols_to_drop_single}")
    df_clean = df_clean.drop(columns=cols_to_drop_single)

# Drop identifier columns
id_cols = ['Prospect ID', 'Lead Number', 'ID']
id_cols_present = [col for col in id_cols if col in df_clean.columns]
if id_cols_present:
    print(f"✓ Dropped identifier columns: {id_cols_present}")
    df_clean = df_clean.drop(columns=id_cols_present)

# Handle missing values
categorical_cols = df_clean.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df_clean[col] = df_clean[col].fillna('Unknown')
print(f"✓ Filled {len(categorical_cols)} categorical columns with 'Unknown'")

numerical_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
print(f"✓ Filled {len(numerical_cols)} numerical columns with median")

# Feature Engineering
if 'Lead Source' in df_clean.columns:
    top_sources = df_clean['Lead Source'].value_counts().nlargest(5).index
    df_clean['Lead Source'] = df_clean['Lead Source'].apply(
        lambda x: x if x in top_sources else 'Others'
    )
    print("✓ Grouped rare Lead Sources")

if 'Lead Origin' in df_clean.columns:
    top_origins = df_clean['Lead Origin'].value_counts().nlargest(5).index
    df_clean['Lead Origin'] = df_clean['Lead Origin'].apply(
        lambda x: x if x in top_origins else 'Others'
    )
    print("✓ Grouped rare Lead Origins")

print(f"\n✓ Preprocessing Complete!")
print(f"  Final shape: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")

# ============================================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("4. EXPLORATORY DATA ANALYSIS (EDA)")
print("="*80)

# Target Distribution
if 'Converted' in df_clean.columns:
    plt.figure(figsize=(10, 6))
    target_counts = df_clean['Converted'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4']
    plt.bar(target_counts.index, target_counts.values, color=colors, edgecolor='black', linewidth=1.5)
    plt.title('Distribution of Target Variable (Converted)', fontsize=14, fontweight='bold')
    plt.xlabel('Converted', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks([0, 1], ['Not Converted', 'Converted'])
    for i, v in enumerate(target_counts.values):
        plt.text(i, v + 50, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('lead_scoring_output/plots/target_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Target distribution plotted")

# Lead Origin vs Conversion
if 'Lead Origin' in df_clean.columns and 'Converted' in df_clean.columns:
    plt.figure(figsize=(14, 7))
    sns.countplot(data=df_clean, x='Lead Origin', hue='Converted', palette=['#FF6B6B', '#4ECDC4'])
    plt.title('Lead Origin vs Conversion Status', fontsize=14, fontweight='bold')
    plt.xlabel('Lead Origin', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Converted', labels=['No', 'Yes'])
    plt.tight_layout()
    plt.savefig('lead_scoring_output/plots/lead_origin_vs_converted.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Lead origin vs conversion plotted")

# Time Spent Analysis
if 'Total Time Spent on Website' in df_clean.columns and 'Converted' in df_clean.columns:
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df_clean, x='Converted', y='Total Time Spent on Website', 
               palette=['#FF6B6B', '#4ECDC4'])
    plt.title('Time Spent on Website vs Conversion', fontsize=14, fontweight='bold')
    plt.xlabel('Converted', fontsize=12)
    plt.ylabel('Total Time Spent (seconds)', fontsize=12)
    plt.xticks([0, 1], ['Not Converted', 'Converted'])
    plt.tight_layout()
    plt.savefig('lead_scoring_output/plots/time_spent_vs_converted.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Time spent analysis plotted")

# Correlation Heatmap
numerical_cols_all = df_clean.select_dtypes(include=['float64', 'int64']).columns
if len(numerical_cols_all) > 1:
    plt.figure(figsize=(12, 10))
    correlation_matrix = df_clean[numerical_cols_all].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
               center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('lead_scoring_output/plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Correlation heatmap plotted")

# Feature Distributions
if len(numerical_cols_all) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(numerical_cols_all[:4]):
        axes[idx].hist(df_clean[col], bins=30, color='#4ECDC4', edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('lead_scoring_output/plots/feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Feature distributions plotted")

# ============================================================================
# 5. MODEL TRAINING
# ============================================================================
print("\n" + "="*80)
print("5. MODEL TRAINING & EVALUATION")
print("="*80)

# Prepare data
X = df_clean.drop(columns=['Converted'])
y = df_clean['Converted']

print(f"\n✓ Data prepared")
print(f"  Features shape: {X.shape}")
print(f"  Target shape: {y.shape}")

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

print(f"  Categorical features: {len(categorical_cols)}")
print(f"  Numerical features: {len(numerical_cols)}")

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✓ Data split")
print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Test set: {X_test.shape[0]} samples")

# Define and train models
print(f"\n✓ Training models...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}
best_model = None
best_score = 0
best_model_name = None

for model_name, model in models.items():
    print(f"\n  Training {model_name}...")
    
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
    
    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1-Score:  {f1:.4f}")
    print(f"    ROC-AUC:   {roc_auc:.4f}")
    
    if accuracy > best_score:
        best_score = accuracy
        best_model = pipeline
        best_model_name = model_name

# ============================================================================
# 6. MODEL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("6. MODEL COMPARISON")
print("="*80)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'F1-Score': [results[m]['f1'] for m in results.keys()],
    'ROC-AUC': [results[m]['roc_auc'] for m in results.keys()]
})

print("\n" + comparison_df.to_string(index=False))

# Visualize comparison
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

for idx, metric in enumerate(metrics):
    axes[idx].bar(comparison_df['Model'], comparison_df[metric], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[idx].set_title(metric, fontweight='bold')
    axes[idx].set_ylim([0, 1])
    axes[idx].tick_params(axis='x', rotation=45)
    for i, v in enumerate(comparison_df[metric]):
        axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('lead_scoring_output/plots/model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✓ Best Model: {best_model_name}")
print(f"  Accuracy: {best_score:.4f}")

# ============================================================================
# 7. BEST MODEL DETAILS
# ============================================================================
print("\n" + "="*80)
print("7. BEST MODEL CLASSIFICATION REPORT")
print("="*80)

y_pred = results[best_model_name]['y_pred']
print(f"\nModel: {best_model_name}")
print(classification_report(y_test, y_pred, target_names=['Not Converted', 'Converted']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
           xticklabels=['Not Converted', 'Converted'],
           yticklabels=['Not Converted', 'Converted'])
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('lead_scoring_output/plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# ROC Curve
y_pred_proba = results[best_model_name]['y_pred_proba']
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = results[best_model_name]['roc_auc']

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#4ECDC4', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title(f'ROC Curve - {best_model_name}', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('lead_scoring_output/plots/roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("8. SAVING RESULTS")
print("="*80)

# Save model
model_path = 'lead_scoring_output/models/best_lead_scoring_model.joblib'
joblib.dump(best_model, model_path)
print(f"✓ Model saved to: {model_path}")

# Save cleaned data
cleaned_data_path = 'lead_scoring_output/cleaned_lead_scoring.csv'
df_clean.to_csv(cleaned_data_path, index=False)
print(f"✓ Cleaned data saved to: {cleaned_data_path}")

# Save comparison
comparison_path = 'lead_scoring_output/model_comparison.csv'
comparison_df.to_csv(comparison_path, index=False)
print(f"✓ Model comparison saved to: {comparison_path}")

# ============================================================================
# 9. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PIPELINE EXECUTION SUMMARY")
print("="*80)

print(f"\n✓ Data Processing:")
print(f"  Original shape: {df.shape}")
print(f"  Cleaned shape: {df_clean.shape}")
print(f"  Rows removed: {df.shape[0] - df_clean.shape[0]}")
print(f"  Columns removed: {df.shape[1] - df_clean.shape[1]}")

print(f"\n✓ Model Training:")
print(f"  Models trained: {len(models)}")
print(f"  Best model: {best_model_name}")
print(f"  Best accuracy: {best_score:.4f}")

print(f"\n✓ Test Set Performance:")
print(f"  Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"  Precision: {results[best_model_name]['precision']:.4f}")
print(f"  Recall: {results[best_model_name]['recall']:.4f}")
print(f"  F1-Score: {results[best_model_name]['f1']:.4f}")
print(f"  ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")

print(f"\n✓ Output Files:")
print(f"  Model: lead_scoring_output/models/best_lead_scoring_model.joblib")
print(f"  Cleaned data: lead_scoring_output/cleaned_lead_scoring.csv")
print(f"  Comparison: lead_scoring_output/model_comparison.csv")
print(f"  Plots: lead_scoring_output/plots/")

print("\n" + "="*80)
print("✓ PIPELINE COMPLETE!")
print("="*80 + "\n")

# Store results for later use
print("✓ Results stored in variables:")
print("  - best_model: The trained model")
print("  - df_clean: Cleaned dataset")
print("  - comparison_df: Model comparison metrics")
print("  - results: Detailed results for each model")

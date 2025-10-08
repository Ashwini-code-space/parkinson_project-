"""
parkinson_ml_pipeline_fixed.py

Full ML pipeline for Parkinson dataset:
- Preprocessing: drop IDs, encode categoricals (Sex -> dummies), impute missing values
- Models: many sklearn classifiers (+ optional XGBoost/LightGBM if installed)
- Evaluation: accuracy, precision, recall, f1, ROC AUC, confusion matrices, classification reports
- Plots: confusion matrices, ROC curves, PR curves, feature importances, learning curve
- Saves models, plots and a CSV summary into ml_results/
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             confusion_matrix, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import joblib

# Optional: try to import XGBoost / LightGBM
has_xgb = False
has_lgb = False
try:
    from xgboost import XGBClassifier
    has_xgb = True
except Exception:
    pass

try:
    from lightgbm import LGBMClassifier
    has_lgb = True
except Exception:
    pass

# ---------------------------
# Config / paths
# ---------------------------
DATA_CSV = "Final_Parkinson_Dataset.csv"   # change if needed
OUTPUT_DIR = "ml_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_csv(DATA_CSV)
print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns from {DATA_CSV}")
print("Columns:", df.columns.tolist())

# ---------------------------
# Preprocessing
# ---------------------------

# Ensure 'Label' exists
if "Label" not in df.columns:
    raise ValueError("Dataset must contain a 'Label' column with values like 'HC' and 'PwPD'.")

# Drop identifier columns that are not features
drop_cols = []
for c in ["filename", "Filename", "source", "Source", "ID"]:
    if c in df.columns:
        drop_cols.append(c)

# Remove drop columns from df copy (keep a copy for reference)
df_proc = df.copy().drop(columns=drop_cols, errors="ignore")
print("After dropping identifiers, columns:", df_proc.columns.tolist())

# Convert numeric-like columns to numeric (Age might be object)
for col in df_proc.columns:
    if col not in ["Label"]:  # don't touch target
        # try convert to numeric if possible
        if df_proc[col].dtype == object:
            # Leave pure string categorical for now (will be dummified)
            pass
        else:
            # ensure numeric dtype
            try:
                df_proc[col] = pd.to_numeric(df_proc[col], errors="coerce")
            except Exception:
                pass

# Identify categorical (object) feature columns (excluding Label)
cat_cols = df_proc.select_dtypes(include=["object"]).columns.tolist()
cat_cols = [c for c in cat_cols if c != "Label"]
print("Categorical columns to encode:", cat_cols)

# One-hot encode categorical feature columns (like 'Sex')
if len(cat_cols) > 0:
    df_proc = pd.get_dummies(df_proc, columns=cat_cols, drop_first=True)
    print("Columns after get_dummies:", df_proc.columns.tolist())

# Convert Label to numeric target using LabelEncoder
le = LabelEncoder()
y = le.fit_transform(df_proc["Label"].astype(str))
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Label mapping:", label_mapping)

# Drop Label column from features
X = df_proc.drop(columns=["Label"], errors="ignore")

# Impute missing numeric values with column means
if X.isna().any().any():
    print("Missing values detected in features; imputing with column means.")
    X = X.fillna(X.mean())

# Ensure all features are numeric now
non_numeric_after = X.select_dtypes(include=["object"]).columns.tolist()
if non_numeric_after:
    raise ValueError(f"Non-numeric columns still present in X after preprocessing: {non_numeric_after}")

print(f"Final feature shape: {X.shape}; target shape: {y.shape}")

# Save feature names for later (ordering)
feature_cols = X.columns.tolist()

# ---------------------------
# Train-test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# ---------------------------
# Helper: pipeline maker
# ---------------------------
def make_pipeline(clf):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf)
    ])

# ---------------------------
# Models to run
# ---------------------------
models = {
    "LogisticRegression": make_pipeline(LogisticRegression(solver="liblinear", random_state=RANDOM_STATE)),
    "KNN": make_pipeline(KNeighborsClassifier(n_neighbors=5)),
    "SVC": make_pipeline(SVC(probability=True, kernel="rbf", random_state=RANDOM_STATE)),
    "DecisionTree": make_pipeline(DecisionTreeClassifier(random_state=RANDOM_STATE)),
    "RandomForest": make_pipeline(RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
    "ExtraTrees": make_pipeline(ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE)),
    "GradientBoosting": make_pipeline(GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE)),
    "AdaBoost": make_pipeline(AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    "GaussianNB": make_pipeline(GaussianNB()),
    "MLP": make_pipeline(MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=RANDOM_STATE))
}

if has_xgb:
    models["XGBoost"] = make_pipeline(XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE))
if has_lgb:
    models["LightGBM"] = make_pipeline(LGBMClassifier(random_state=RANDOM_STATE))

# ---------------------------
# Run training & evaluation
# ---------------------------
results = []

# Setup ROC main figure
plt.figure(figsize=(10, 8))
plt.title("ROC Curves")
mean_fpr = np.linspace(0, 1, 100)

for name, pipeline in models.items():
    print(f"\n=== Training {name} ===")
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)
    # get probabilities or decision scores
    clf = pipeline.named_steps['clf']
    if hasattr(clf, "predict_proba"):
        try:
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            # fallback to decision_function if predict_proba fails
            try:
                y_prob = pipeline.decision_function(X_test)
            except Exception:
                y_prob = None
    else:
        try:
            y_prob = pipeline.decision_function(X_test)
        except Exception:
            y_prob = None

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob) if (y_prob is not None and len(np.unique(y_test)) > 1) else np.nan

    print(f"{name} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ROC AUC: {roc_auc:.4f}")

    results.append({
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc
    })

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", ax=ax_cm)
    ax_cm.set_title(f"{name} Confusion Matrix")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    fig_cm.tight_layout()
    fig_cm.savefig(os.path.join(OUTPUT_DIR, f"{name}_confusion_matrix.png"))
    plt.close(fig_cm)

    # Classification report
    creport = classification_report(y_test, y_pred, zero_division=0)
    with open(os.path.join(OUTPUT_DIR, f"{name}_classification_report.txt"), "w") as fh:
        fh.write(creport)

    # ROC
    if y_prob is not None and len(np.unique(y_test)) > 1:
        try:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, lw=1, label=f"{name} (AUC={roc_auc:.3f})")
        except Exception as e:
            print(f"Could not compute ROC for {name}: {e}")

    # Precision-Recall curve
    if y_prob is not None:
        try:
            p, r, _ = precision_recall_curve(y_test, y_prob)
            fig_pr, ax_pr = plt.subplots(figsize=(5, 4))
            ax_pr.plot(r, p)
            ax_pr.set_xlabel("Recall")
            ax_pr.set_ylabel("Precision")
            ax_pr.set_title(f"{name} Precision-Recall")
            fig_pr.tight_layout()
            fig_pr.savefig(os.path.join(OUTPUT_DIR, f"{name}_pr_curve.png"))
            plt.close(fig_pr)
        except Exception as e:
            print(f"Could not compute PR curve for {name}: {e}")

    # Feature importances (if available)
    try:
        # extract inner estimator (clf) for tree-based models
        estimator = pipeline.named_steps["clf"]
        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
            fi_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
            fi_df = fi_df.sort_values("importance", ascending=False).head(30)
            fig_fi, ax_fi = plt.subplots(figsize=(8, 6))
            ax_fi.barh(fi_df["feature"][::-1], fi_df["importance"][::-1])
            ax_fi.set_title(f"{name} Top Features")
            fig_fi.tight_layout()
            fig_fi.savefig(os.path.join(OUTPUT_DIR, f"{name}_feature_importance.png"))
            plt.close(fig_fi)
    except Exception as e:
        print(f"Feature importance error for {name}: {e}")

    # Save model
    joblib.dump(pipeline, os.path.join(OUTPUT_DIR, f"{name}_model.joblib"))

# Finalize and save ROC plot
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - All Models")
plt.legend(loc="lower right", fontsize="small")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "all_models_roc.png"))
plt.clf()
plt.close()

# Save metrics table
results_df = pd.DataFrame(results).sort_values(by="f1", ascending=False)
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_results.csv"), index=False)
print("\nModel comparison saved to:", os.path.join(OUTPUT_DIR, "model_comparison_results.csv"))
print(results_df)

# Summary metric plots
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
# Replace NaN with 0 for plotting stability
plot_df = results_df.copy().fillna(0)
fig_sum, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
for i, m in enumerate(metrics):
    ax = axes[i]
    sns.barplot(x="model", y=m, data=plot_df, ax=ax)
    ax.set_xticklabels(plot_df["model"], rotation=45, ha="right")
    ax.set_title(m)
fig_sum.tight_layout()
fig_sum.savefig(os.path.join(OUTPUT_DIR, "metrics_summary.png"))
plt.close(fig_sum)

# Learning curve for top model
if not results_df.empty:
    top_model_name = results_df.iloc[0]["model"]
    top_model_path = os.path.join(OUTPUT_DIR, f"{top_model_name}_model.joblib")
    if os.path.exists(top_model_path):
        top_pipeline = joblib.load(top_model_path)
        try:
            train_sizes, train_scores, test_scores = learning_curve(
                top_pipeline, X, y, cv=StratifiedKFold(n_splits=CV_FOLDS),
                scoring="f1", train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1)
            train_scores_mean = np.mean(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)

            fig_lc, ax_lc = plt.subplots(figsize=(7, 6))
            ax_lc.plot(train_sizes, train_scores_mean, label="Training score")
            ax_lc.plot(train_sizes, test_scores_mean, label="Cross-val score")
            ax_lc.set_xlabel("Training examples")
            ax_lc.set_ylabel("F1 score")
            ax_lc.set_title(f"Learning curve - {top_model_name}")
            ax_lc.legend(loc="best")
            fig_lc.tight_layout()
            fig_lc.savefig(os.path.join(OUTPUT_DIR, f"{top_model_name}_learning_curve.png"))
            plt.close(fig_lc)
        except Exception as e:
            print(f"Could not compute learning curve for {top_model_name}: {e}")

print("\nAll results, models and plots saved to:", OUTPUT_DIR)

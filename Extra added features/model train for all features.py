import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, AdaBoostClassifier,
                              BaggingClassifier, VotingClassifier, StackingClassifier)
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
TRAIN_PD_CSV = "output_training_dataset_pd.csv"  # Fixed training PD file
TRAIN_HEALTHY_CSV = "output_training_dataset_healthy.csv"  # Fixed training healthy file
OUTPUT_DIR = "ml_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42
CV_FOLDS = 5

# ---------------------------
# Preprocessing function (reusable for train/test)
# ---------------------------
def preprocess_df(df, is_train=True, le=None):
    # Drop identifier columns
    drop_cols = []
    for c in ["Sample", "filename", "Filename", "source", "Source", "ID"]:
        if c in df.columns:
            drop_cols.append(c)
    # Keep a copy of identifiers if present (for predictions)
    if "Sample" in df.columns:
        identifiers = df["Sample"].copy()
    else:
        identifiers = pd.Series(range(len(df)), name="Index")  # Fallback
    df = df.drop(columns=drop_cols, errors="ignore")
    # Convert numeric-like columns to numeric
    for col in df.columns:
        if col not in ["Label"]:
            if df[col].dtype == object:
                pass
            else:
                try:
                    df[col] = pd.to_numeric(df[col], errors="ignore")
                except Exception:
                    pass
    # Identify categorical columns (exclude Label)
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != "Label"]
    # One-hot encode categoricals (e.g., 'Sex')
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    # Impute missing numeric values with means
    if df.isna().any().any():
        df = df.fillna(df.mean())
    # Handle Label (only for training)
    if is_train:
        if "Label" in df.columns:
            le = LabelEncoder()
            y = le.fit_transform(df["Label"].astype(str))
            df = df.drop(columns=["Label"], errors="ignore")
        else:
            raise ValueError("Training data must contain 'Label' column.")
    else:
        y = None  # No label in test
    # Ensure all are numeric
    non_numeric = df.select_dtypes(include=["object"]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric columns in DataFrame: {non_numeric}")
    return df, y, identifiers, le if is_train else None

# ---------------------------
# Load and combine training data
# ---------------------------
print("Loading training data...")
df_train_pd = pd.read_csv(TRAIN_PD_CSV)
df_train_healthy = pd.read_csv(TRAIN_HEALTHY_CSV)
# Combine
df_train = pd.concat([df_train_pd, df_train_healthy], ignore_index=True)
print(f"Combined training: {df_train.shape[0]} rows, {df_train.shape[1]} columns")
# Preprocess training
X_train, y_train, _, le = preprocess_df(df_train, is_train=True)
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Label mapping:", label_mapping)
feature_cols = X_train.columns.tolist()  # Save for alignment
print(f"Training features: {X_train.shape}; targets: {y_train.shape}")

# ---------------------------
# Prompt for test file
# ---------------------------
test_file = input("Enter the path to the test file (CSV or XLSX): ").strip()
if not os.path.exists(test_file):
    raise FileNotFoundError(f"Test file not found: {test_file}")
print(f"Loading test file: {test_file}")
if test_file.lower().endswith(".csv"):
    df_test = pd.read_csv(test_file)
elif test_file.lower().endswith(".xlsx"):
    df_test = pd.read_excel(test_file, sheet_name="Sheet1")
else:
    raise ValueError("Test file must be .csv or .xlsx")
print(f"Loaded test: {df_test.shape[0]} rows, {df_test.shape[1]} columns")
# Preprocess test (no labels)
X_test, _, test_identifiers, _ = preprocess_df(df_test, is_train=False)
# Align columns to training (add missing with 0, drop extra)
missing_cols = set(feature_cols) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0
extra_cols = set(X_test.columns) - set(feature_cols)
X_test = X_test.drop(columns=extra_cols, errors="ignore")
X_test = X_test[feature_cols]  # Reorder to match train
print(f"Test features: {X_test.shape}")

# ---------------------------
# Helper: pipeline maker
# ---------------------------
def make_pipeline(clf, use_feature_selection=False):
    steps = [("scaler", StandardScaler())]
    if use_feature_selection:
        steps.append(("feature_selection", SelectKBest(score_func=f_classif, k=10)))
    steps.append(("clf", clf))
    return Pipeline(steps)

# ---------------------------
# Models to run (expanded with recommended combinations)
# ---------------------------
base_models = [
    ("LR", LogisticRegression(solver="liblinear", random_state=RANDOM_STATE)),
    ("KNN", KNeighborsClassifier(n_neighbors=5)),
    ("SVC", SVC(probability=True, kernel="rbf", random_state=RANDOM_STATE)),
    ("LinearSVC", LinearSVC(random_state=RANDOM_STATE)),
    ("NuSVC", NuSVC(probability=True, random_state=RANDOM_STATE)),
    ("DT", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ("RF", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
    ("ET", ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE)),
    ("GB", GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE)),
    ("Ada", AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    ("GNB", GaussianNB()),
    ("MLP", MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=RANDOM_STATE)),
    ("SGD", SGDClassifier(random_state=RANDOM_STATE)),
    ("Ridge", RidgeClassifier(random_state=RANDOM_STATE)),
    ("Perceptron", Perceptron(random_state=RANDOM_STATE)),
    ("Bagging", BaggingClassifier(random_state=RANDOM_STATE))
]
if has_xgb:
    base_models.append(("XGB", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE)))
if has_lgb:
    base_models.append(("LGB", LGBMClassifier(random_state=RANDOM_STATE)))

models = {name: make_pipeline(clf) for name, clf in base_models}

# Add recommended ensemble combinations
# 1. Voting_All: Hard voting with all base models
voting_clf = VotingClassifier(estimators=base_models, voting='hard')
models["Voting_All"] = make_pipeline(voting_clf)

# 2. Voting_Selected: Hard voting with LR, SVC, RF
voting_selected = VotingClassifier(estimators=[base_models[0], base_models[2], base_models[6]], voting='hard')  # LR, SVC, RF
models["Voting_Selected"] = make_pipeline(voting_selected)

# 3. Stacking: Base models with predict_proba, LR meta-learner
prob_models = [(name, clf) for name, clf in base_models if name not in ["LinearSVC", "SGD", "Ridge", "Perceptron"]]
stacking_clf = StackingClassifier(estimators=prob_models, final_estimator=LogisticRegression(), cv=5)
models["Stacking"] = make_pipeline(stacking_clf)

# 4. Stacking_RF_Meta: Base models with predict_proba, RF meta-learner
stacking_rf = StackingClassifier(estimators=prob_models, final_estimator=RandomForestClassifier(n_estimators=100), cv=5)
models["Stacking_RF_Meta"] = make_pipeline(stacking_rf)

# 5. Stacking_SVM_RF_XGB: Recommended stacking (SVM, RF, XGB) with LR meta-learner
stacking_svm_rf_xgb_estimators = [
    ("SVC", SVC(probability=True, kernel="rbf", random_state=RANDOM_STATE)),
    ("RF", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))
]
if has_xgb:
    stacking_svm_rf_xgb_estimators.append(("XGB", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE)))
stacking_svm_rf_xgb = StackingClassifier(
    estimators=stacking_svm_rf_xgb_estimators,
    final_estimator=LogisticRegression(),
    cv=5
)
models["Stacking_SVM_RF_XGB"] = make_pipeline(stacking_svm_rf_xgb)

# 6. Stacking_GB_RF: GB (XGB or LGB) + RF with RF meta-learner
stacking_gb_rf_estimators = [
    ("RF", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))
]
if has_xgb:
    stacking_gb_rf_estimators.append(("XGB", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE)))
elif has_lgb:
    stacking_gb_rf_estimators.append(("LGB", LGBMClassifier(random_state=RANDOM_STATE)))
else:
    stacking_gb_rf_estimators.append(("GB", GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE)))
stacking_gb_rf = StackingClassifier(
    estimators=stacking_gb_rf_estimators,
    final_estimator=RandomForestClassifier(n_estimators=100),
    cv=5
)
models["Stacking_GB_RF"] = make_pipeline(stacking_gb_rf)

# 7. Voting_SVM_KNN_GB: Hard voting with SVM, KNN, GB
voting_svm_knn_gb_estimators = [
    ("SVC", SVC(probability=True, kernel="rbf", random_state=RANDOM_STATE)),
    ("KNN", KNeighborsClassifier(n_neighbors=5)),
    ("GB", GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE))
]
voting_svm_knn_gb = VotingClassifier(estimators=voting_svm_knn_gb_estimators, voting='hard')
models["Voting_SVM_KNN_GB"] = make_pipeline(voting_svm_knn_gb)

# 8. Stacking_ANOVA_RF_SVM_MLP: ANOVA feature selection + Stacking (RF, SVM, MLP)
stacking_anova_estimators = [
    ("RF", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
    ("SVC", SVC(probability=True, kernel="rbf", random_state=RANDOM_STATE)),
    ("MLP", MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=RANDOM_STATE))
]
stacking_anova = StackingClassifier(
    estimators=stacking_anova_estimators,
    final_estimator=LogisticRegression(),
    cv=5
)
models["Stacking_ANOVA_RF_SVM_MLP"] = make_pipeline(stacking_anova, use_feature_selection=True)

# 9. Stacking_Bagging_Boosting: Bagging + AdaBoost/XGB with LR meta-learner
stacking_bagging_boosting_estimators = [
    ("Bagging", BaggingClassifier(random_state=RANDOM_STATE)),
    ("Ada", AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE))
]
if has_xgb:
    stacking_bagging_boosting_estimators.append(("XGB", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE)))
stacking_bagging_boosting = StackingClassifier(
    estimators=stacking_bagging_boosting_estimators,
    final_estimator=LogisticRegression(),
    cv=5
)
models["Stacking_Bagging_Boosting"] = make_pipeline(stacking_bagging_boosting)

# ---------------------------
# Train and evaluate models (CV on training data)
# ---------------------------
results = []
predictions_df = pd.DataFrame({"Sample": test_identifiers})
scoring = {
    "accuracy": "accuracy",
    "precision": "precision_weighted",
    "recall": "recall_weighted",
    "f1": "f1_weighted",
    "roc_auc": "roc_auc" if len(np.unique(y_train)) > 1 else None
}
if scoring["roc_auc"] is None:
    del scoring["roc_auc"]  # Skip if only one class

for name, pipeline in models.items():
    print(f"\n=== Training {name} ===")
    # Train on full training data
    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        print(f"Training error for {name}: {e}")
        results.append({
            "model": name,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "roc_auc": np.nan
        })
        predictions_df[f"{name}_pred"] = ["Error"] * len(test_identifiers)
        continue
    # Cross-validation on training data
    current_scoring = scoring.copy()
    # Skip roc_auc for models without predict_proba
    if name in ["LinearSVC", "SGD", "Ridge", "Perceptron", "Voting_All", "Voting_Selected", "Voting_SVM_KNN_GB"]:
        current_scoring.pop("roc_auc", None)
    try:
        cv_scores = cross_validate(
            pipeline, X_train, y_train,
            cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            scoring=current_scoring, return_train_score=False
        )
        # Compute mean of CV scores
        acc = cv_scores["test_accuracy"].mean()
        prec = cv_scores["test_precision"].mean()
        rec = cv_scores["test_recall"].mean()
        f1 = cv_scores["test_f1"].mean()
        roc_auc = cv_scores.get("test_roc_auc", np.array([np.nan])).mean()
        print(f"{name} CV Metrics | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ROC AUC: {roc_auc:.4f}")
        results.append({
            "model": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc
        })
    except Exception as e:
        print(f"Error in CV for {name}: {e}")
        results.append({
            "model": name,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "roc_auc": np.nan
        })
    # Predict on test
    try:
        y_pred_num = pipeline.predict(X_test)
        y_pred = le.inverse_transform(y_pred_num)
        predictions_df[f"{name}_pred"] = y_pred
    except Exception as e:
        print(f"Prediction error for {name}: {e}")
        predictions_df[f"{name}_pred"] = ["Error"] * len(test_identifiers)
    # Feature importances (if available)
    try:
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
    try:
        joblib.dump(pipeline, os.path.join(OUTPUT_DIR, f"{name}_model.joblib"))
    except Exception as e:
        print(f"Error saving model {name}: {e}")

# Save predictions DF
predictions_df.to_csv(os.path.join(OUTPUT_DIR, "all_models_predictions.csv"), index=False)
print("\nPredictions per model saved to:", os.path.join(OUTPUT_DIR, "all_models_predictions.csv"))
print(predictions_df.head())  # Print sample

# Save CV metrics table
results_df = pd.DataFrame(results).sort_values(by="f1", ascending=False, na_position="last")
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_results.csv"), index=False)
print("\nCV model comparison saved to:", os.path.join(OUTPUT_DIR, "model_comparison_results.csv"))
print(results_df)

# Summary metric plots
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
plot_df = results_df.copy().fillna(0)
fig_sum, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
for i, m in enumerate(metrics):
    ax = axes[i]
    sns.barplot(x="model", y=m, data=plot_df, ax=ax)
    ax.set_xticklabels(plot_df["model"], rotation=45, ha="right")
    ax.set_title(f"CV {m}")
fig_sum.tight_layout()
fig_sum.savefig(os.path.join(OUTPUT_DIR, "cv_metrics_summary.png"))
plt.close(fig_sum)

# Learning curve for top model
if not results_df.empty:
    top_model_name = results_df.iloc[0]["model"]
    top_model_path = os.path.join(OUTPUT_DIR, f"{top_model_name}_model.joblib")
    if os.path.exists(top_model_path):
        top_pipeline = joblib.load(top_model_path)
        try:
            train_sizes, train_scores, test_scores = learning_curve(
                top_pipeline, X_train, y_train, cv=StratifiedKFold(n_splits=CV_FOLDS),
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

print("\nAll results, models, predictions, and plots saved to:", OUTPUT_DIR)

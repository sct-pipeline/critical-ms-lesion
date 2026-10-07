"""
Step 5 of the critical lesion classification study: train a classifier of critical vs non-critical
spinal cord lesions, compare several model families, and analyse which features drive the decision.
The protocol is the one of detection/analyze_reports.py: a subject-based nested cross-validation with
no held-out set, the inner loop estimating the hyperparameters with a Bayesian search, and a final
model fitted on the entire dataset from which the top predictive features are extracted.

Two arms are trained and evaluated on the same subject-wise folds, so that they can be compared:
    auto    features of the predicted lesion segmentations, i.e. the situation at inference time
    manual  features of the manual lesion segmentations, i.e. the upper bound with perfect masks

Each arm trains and tests on its own mask source, so no model is trained on one kind of mask and
tested on the other. The auto arm keeps the predicted lesions overlapping no manual lesion, labelled
non-critical, so it carries the false positives of the segmentation. Only the subjects present in
both tables are used, and both arms share the same folds and the same feature columns.

The run has two passes:
    1. every model family of MODEL_CONFIGS is nested-cross-validated on every arm, all of them on the
       same folds and the same features, which gives the comparison table
    2. the best model of each arm (highest mean F1) is fitted on the entire dataset, its top features
       are extracted, and the same nested cross-validation with the same search is run again on those
       features alone

Beware when reporting pass 1: the model family is chosen on the same cross-validation that scores it,
so the winner's score is optimistic. The same holds for pass 2, whose features were ranked with the
labels of every subject. Both are exploratory; the honest figure for a single model is its own row of
the comparison table.

Missing values (e.g. the standard deviation of a measure over the slices of a single-slice lesion) are
replaced by the mean of their feature, and the linear models are standardised. Both steps sit inside
the pipeline, so they are fitted on the training folds only. Results are reported at three levels:
    lesion      critical vs non-critical, over the lesions of each arm
    patient     whether a patient's highest-scoring lesion is one of its critical lesions, counted
                over the patients holding a critical lesion in the manual segmentation, so that a
                patient whose critical lesion the segmentation missed counts as an error rather than
                dropping out of the comparison
    end to end  of the critical lesions of the manual segmentations, how many were both found by the
                segmentation and classified critical by the auto arm (needs --critical-lesions)

Outputs, in the output folder:
    folds.csv                     fold of every subject
    fold_results.csv              lesion-level performance of every arm, model and fold
    results_summary.csv           lesion-level performance of every arm and model, mean and std
    patient_level.csv             patient-level performance of every arm and model
    end_to_end.csv                end-to-end sensitivity of every auto model (--critical-lesions)
    predictions.csv               per-lesion out-of-fold probability, for error analysis
    roc_pr_calibration_all_features.png       ROC, precision-recall and calibration curves of every
                                              run on the full feature set
    roc_pr_calibration_top_<n>_features.png   the same three curves for the top-features runs, kept
                                              apart since they are not comparable to the full ones
    top_features_<arm>_<model>.csv         features of the final model, ranked
    shap_beeswarm_<arm>_<model>.png        SHAP summary plot (tree models only)
    shap_bar_<arm>_<model>.png             mean absolute SHAP value of the top features (tree models)
    feature_importance_<arm>_<model>.png   the ranking as a bar plot
    train_classifier.log          full report

Input:
    -f / --features-folder: folder holding features_manual.csv and features_pred.csv (output of extract_features.py)
    -o: path to the output folder where the results will be saved
    --critical-lesions: path to critical_lesions.csv (output of evaluate_segmentation.py); when given,
        the end-to-end sensitivity for critical lesions is reported
    --models: model families to compare (default: all of them). Every family costs one Bayesian search
        per arm and per fold, so the full comparison takes a while; pass a subset to iterate quickly.
    --features: which feature variant to use (default: all). The level-type measures (cord shape and
        asymmetry) have mean, min and max correlated above 0.95, so their min and max are redundant,
        while the tract measures do not (min and max correlate 0.66) and keep all four aggregates.
    --select-k: when given, every model family is also run with a SelectKBest step inside the
        pipeline, so the features are chosen on each training fold alone. Either a number of features
        (e.g. 10) or "tune" to let the inner search pick it. This is the unbiased counterpart of the
        second pass below, whose features are ranked on a model fitted to every subject.
    --n-splits: number of outer cross-validation folds (default: 5)
    --n-repeats: number of times the whole cross-validation is repeated with a different shuffle
        (default: 1). Repeating stabilises the mean, which is what makes two variants comparable.
    --n-splits-inner: number of inner folds of the hyperparameter search (default: 3)
    --n-iter: number of hyperparameter settings sampled by the Bayesian search (default: 20)
    --n-top-features: number of features kept in the second pass (default: 10)
    --keep-norm-diff: keep the (diff_with_hc_norm) features, which are dropped by default

Author: Pierre-Louis Benveniste
"""
import os
import sys
import shlex
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from loguru import logger
from tqdm import tqdm
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.calibration import CalibrationDisplay
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from xgboost import XGBClassifier

from include_io import METADATA_COLUMNS


RANDOM_STATE = 42
# Feature table of each arm, the auto arm being the one that matters
ARM_TABLES = {"auto": "features_pred.csv", "manual": "features_manual.csv"}
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]
# The six composite tract measures of the feature tables, which the 36 individual PAM50 tracts add
# up to. Named here rather than imported, to keep this script independent of detection/.
TRACT_COMPOSITES = ["total % (all tracts)", "white matter", "gray matter", "dorsal columns",
                    "lateral funiculi", "ventral funiculi"]
# Feature variants, from the redundancy measured on the feature tables:
#   all               every feature (409)
#   no_minmax_level   without the min and max aggregates of the level-type measures (289)
#   composite_tracts  the above, and only the composite tract measures, not the 36 PAM50 ones (145)
FEATURE_VARIANTS = ["all", "no_minmax_level", "composite_tracts"]
# Model families compared on the same folds and features. "scale" standardises the features, which the
# linear models need and the tree ensembles do not care about.
MODEL_CONFIGS = {
    "gradient_boosting": {
        "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "search_space": {
            "clf__n_estimators": Integer(50, 400),
            "clf__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            "clf__max_depth": Integer(2, 10),
            "clf__subsample": Real(0.5, 1.0),
        },
    },
    "xgboost": {
        "estimator": XGBClassifier(random_state=RANDOM_STATE, n_jobs=1),
        "search_space": {
            "clf__n_estimators": Integer(50, 500),
            "clf__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            "clf__max_depth": Integer(2, 8),
            "clf__min_child_weight": Integer(1, 10),
            "clf__subsample": Real(0.5, 1.0),
            "clf__colsample_bytree": Real(0.3, 1.0),
            "clf__reg_lambda": Real(1e-3, 1e2, prior="log-uniform"),
        },
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        "search_space": {
            "clf__n_estimators": Integer(100, 500),
            "clf__max_depth": Integer(2, 20),
            "clf__min_samples_split": Integer(2, 10),
            "clf__max_features": Real(0.05, 1.0),
        },
    },
    "logistic_regression": {
        "estimator": LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000, random_state=RANDOM_STATE),
        "search_space": {"clf__C": Real(1e-3, 1e2, prior="log-uniform")},
        "scale": True,
    },
    "linear_svm": {
        "estimator": SVC(kernel="linear", probability=True, random_state=RANDOM_STATE),
        "search_space": {"clf__C": Real(1e-3, 1e2, prior="log-uniform")},
        "scale": True,
    },
}


def selection_size(value):
    """Type of --select-k: either the string "tune" or a positive number of features."""
    if value == "tune":
        return value
    size = int(value)
    if size < 1:
        raise argparse.ArgumentTypeError("--select-k must be 'tune' or a positive integer")
    return size


def parse_args():
    parser = argparse.ArgumentParser(description="Compare model families on the features of the predicted and of the manual lesion segmentations, with a subject-wise nested cross-validation.")
    parser.add_argument("-f", "--features-folder", type=str, required=True, help="Folder holding features_manual.csv and features_pred.csv (output of extract_features.py).")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder where the results will be saved.")
    parser.add_argument("--critical-lesions", type=str, help="Path to critical_lesions.csv (output of evaluate_segmentation.py). When given, the end-to-end sensitivity for critical lesions is reported, counting the lesions the segmentation missed as errors.")
    parser.add_argument("--models", type=str, nargs="+", default=list(MODEL_CONFIGS), choices=list(MODEL_CONFIGS), help="Model families to compare (default: all of them).")
    parser.add_argument("--features", type=str, default="all", choices=FEATURE_VARIANTS, help="Feature variant to use (default: all). no_minmax_level drops the redundant min and max aggregates of the level-type measures; composite_tracts also keeps only the composite tract measures.")
    parser.add_argument("--select-k", type=selection_size, help="Also run every model family with a SelectKBest step inside the pipeline, fitted on each training fold alone: a number of features (e.g. 10), or 'tune' to let the inner search pick it. This is the unbiased counterpart of the top-features pass.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of outer cross-validation folds (default: 5).")
    parser.add_argument("--n-repeats", type=int, default=1, help="Number of times the cross-validation is repeated with a different shuffle (default: 1).")
    parser.add_argument("--n-splits-inner", type=int, default=3, help="Number of inner folds of the hyperparameter search (default: 3).")
    parser.add_argument("--n-iter", type=int, default=20, help="Number of hyperparameter settings sampled by the Bayesian search (default: 20).")
    parser.add_argument("--n-top-features", type=int, default=10, help="Number of features kept in the second pass (default: 10).")
    parser.add_argument("--keep-norm-diff", action="store_true", help="Keep the (diff_with_hc_norm) features, which are dropped by default.")
    return parser.parse_args()


def is_tract_feature(column):
    """Whether a feature column measures the involvement of a tract, rather than the cord itself."""
    return column.startswith("PAM50_") or any(column.startswith(f"{name}_") for name in TRACT_COMPOSITES)


def select_feature_variant(columns, variant):
    """
    Restrict the feature columns to one of FEATURE_VARIANTS. The min and max aggregates are dropped
    only for the level-type measures (cord shape and asymmetry), where mean, min and max correlate
    above 0.95; the tract measures keep all four, their min and max correlating only 0.66, min being
    the floor of the involvement along the lesion and max its worst slice.
    """
    if variant == "all":
        return columns
    columns = [column for column in columns if is_tract_feature(column) or not column.endswith(("_min", "_max"))]
    if variant == "composite_tracts":
        columns = [column for column in columns if not column.startswith("PAM50_")]
    return columns


def get_feature_columns(tables, keep_norm_diff, variant="all"):
    """
    Feature columns shared by every arm, without the metadata columns, the redundant
    (diff_with_hc_norm) features unless asked for, and the features that are constant in any arm
    (they carry no information and break the SHAP analysis). Both arms use the same columns, so that
    their performances can be compared.
    """
    columns = [column for column in tables["manual"].columns if column not in METADATA_COLUMNS]
    for df in tables.values():
        columns = [column for column in columns if column in df.columns and df[column].nunique() > 1]
    if not keep_norm_diff:
        columns = [column for column in columns if "diff_with_hc_norm" not in column]
    return select_feature_variant(columns, variant)


def get_xy(df, feature_columns):
    """
    Feature matrix and labels of a feature table. Infinite values (produced by the ratio-based
    asymmetry measures when a quadrant area is zero) become missing values, imputed by the pipeline.
    """
    X = df[feature_columns].replace([np.inf, -np.inf], np.nan).astype(float)
    return X, df["critical_lesion_label"].astype(int)


def get_folds(df_reference, n_splits, n_repeats=1):
    """
    Subject-wise cross-validation folds, stratified on the lesion label of the reference table, and
    optionally repeated with a different shuffle each time. A fold is a set of subjects rather than
    row indices, so that every arm and every model is trained and evaluated on exactly the same
    patients. No subject is set aside: within a repeat, each one is evaluated exactly once.
    Output:
        A list of (repeat, fold, subjects) tuples
    """
    subjects = df_reference["subject"]
    folds = []
    for repeat in range(n_repeats):
        # The first repeat keeps RANDOM_STATE, so a single-repeat run reproduces earlier results
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE + repeat)
        for fold, (_, evaluation_idx) in enumerate(
            splitter.split(df_reference, df_reference["critical_lesion_label"], subjects), start=1
        ):
            folds.append((repeat + 1, fold, set(subjects.iloc[evaluation_idx])))
    return folds


def build_pipeline(config, select_k=None):
    """
    Mean imputation, standardisation for the models that need it, an optional feature selection, then
    the classifier. The selector sits in the pipeline on purpose: it is then fitted on the training
    data of every split and never sees the labels of the data it is evaluated on.
    Input:
        select_k: number of features to keep, "tune" to let the search decide, or None for no selection
    """
    steps = [("imputer", SimpleImputer(strategy="mean"))]
    if config.get("scale"):
        steps.append(("scaler", StandardScaler()))
    if select_k is not None:
        steps.append(("selector", SelectKBest(f_classif, k="all" if select_k == "tune" else select_k)))
    steps.append(("clf", clone(config["estimator"])))
    return Pipeline(steps)


def fit_model(config, X, y, groups, n_splits_inner, n_iter, select_k=None):
    """
    Fit one model family, its hyperparameters picked by a Bayesian search over an inner subject-wise
    cross-validation. Imputation, standardisation and feature selection are inside the pipeline,
    hence fitted on the inner training folds only.
    Input:
        select_k: number of features to keep, "tune" to add it to the search, or None for no selection
    Output:
        (best_estimator, best_params) with the estimator being the fitted pipeline
    """
    search_space = dict(config["search_space"])
    if select_k == "tune":
        search_space["selector__k"] = Integer(5, min(100, X.shape[1]))
    search = BayesSearchCV(
        build_pipeline(config, select_k), search_space, cv=GroupKFold(n_splits_inner),
        n_iter=n_iter, scoring="f1", n_jobs=-1, random_state=RANDOM_STATE,
    )
    search.fit(X, y, groups=groups)
    return search.best_estimator_, dict(search.best_params_)


def compute_metrics(y_true, y_prob):
    """Classification metrics of one fold. ROC-AUC and AP are undefined on a single-class fold."""
    y_pred = (y_prob >= 0.5).astype(int)
    both_classes = y_true.nunique() > 1
    return {
        "n_lesions": len(y_true),
        "n_critical": int(y_true.sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if both_classes else np.nan,
        "average_precision": average_precision_score(y_true, y_prob) if both_classes else np.nan,
    }


def run_nested_cv(df, feature_columns, folds, config, arm, model_name, n_splits_inner, n_iter, select_k=None):
    """
    Nested cross-validation of one model family on one arm: each fold in turn is the evaluation set,
    the model being fitted on the other folds with its hyperparameters chosen by a Bayesian search
    over an inner subject-wise cross-validation of those folds. Every subject is evaluated once and
    nothing is held out.
    Output:
        (fold_rows, predictions), the per-fold metrics and the out-of-fold predictions
    """
    fold_rows, predictions = [], []
    for repeat, fold, fold_subjects in tqdm(folds, desc=f"{arm} / {model_name}"):
        is_evaluated = df["subject"].isin(fold_subjects)
        df_train, df_eval = df[~is_evaluated], df[is_evaluated]
        if df_eval.empty or df_train["critical_lesion_label"].nunique() < 2:
            logger.warning(f"{arm} / {model_name}, repeat {repeat} fold {fold}: skipped, nothing to evaluate or a single class in training")
            continue

        X_train, y_train = get_xy(df_train, feature_columns)
        model, best_params = fit_model(config, X_train, y_train, df_train["subject"], n_splits_inner, n_iter, select_k)
        logger.info(f"{arm} / {model_name}, repeat {repeat} fold {fold}: best parameters {best_params}")

        X_eval, y_eval = get_xy(df_eval, feature_columns)
        y_prob = model.predict_proba(X_eval)[:, 1]
        fold_rows.append({"arm": arm, "model": model_name, "repeat": repeat, "fold": fold, **compute_metrics(y_eval, y_prob)})
        # matched only exists for the predicted lesions: it is left empty for the manual arm
        predictions.append(
            df_eval.reindex(columns=["subject", "scan_id", "lesion_label", "matched"])
            .assign(arm=arm, model=model_name, repeat=repeat, fold=fold, y_true=y_eval.values,
                    y_prob=y_prob, y_pred=(y_prob >= 0.5).astype(int))
        )

    return fold_rows, predictions


def summarise(df_folds):
    """
    Lesion counts summed and metrics averaged over the folds of every arm and model. The standard
    error is what tells two runs apart: repeating the cross-validation leaves the fold-to-fold
    standard deviation where it is and shrinks the error on the mean.
    Output:
        A dataframe indexed by (arm, model)
    """
    by_run = df_folds.groupby(["arm", "model"], sort=False)
    stats = by_run[METRICS].agg(["mean", "std", "sem"])
    stats.columns = [f"{metric}_{statistic}" for metric, statistic in stats.columns]
    # Lesion counts are summed over the folds of one repeat, not over the repeats
    counts = by_run[["n_lesions", "n_critical"]].sum().div(df_folds["repeat"].nunique()).round().astype(int)
    return counts.join(stats)


def format_summary(df_folds):
    """The same summary as a table of 'mean ± std' strings, for the log."""
    by_run = df_folds.groupby(["arm", "model"], sort=False)
    n_repeats = df_folds["repeat"].nunique()
    counts = by_run[["n_lesions"]].sum().div(n_repeats).round().astype(int)
    return counts.join(by_run[METRICS].agg(lambda values: f"{values.mean():.3f} ± {values.std():.3f}"))


def top_lesion_accuracy(df_predictions, critical_subjects):
    """
    Patient-level check: among the patients holding at least one critical lesion in the manual
    segmentation, how often the lesion with the highest predicted probability is a critical one.
    The denominator is the same for every arm and model, so a patient whose critical lesion the
    segmentation missed counts as an error instead of dropping out of the comparison. The count is
    averaged over the repeats of the cross-validation, in which a patient can go either way.
    Output:
        (n_subjects, mean n_correct over the repeats)
    """
    df = df_predictions[df_predictions["subject"].isin(critical_subjects)]
    df_top = df.loc[df.groupby(["repeat", "subject"])["y_prob"].idxmax()]
    return len(critical_subjects), df_top["y_true"].sum() / df_predictions["repeat"].nunique()


def end_to_end_sensitivity(critical_lesions_csv, df_predictions):
    """
    End-to-end sensitivity for critical lesions: of the critical lesions of the manual segmentations,
    how many were both found by the segmentation and classified critical. A lesion the segmentation
    missed counts as an error, since the classifier never gets to see it. A lesion that was found
    counts as recovered when a critical predicted lesion of its scan was classified critical, which
    is exact unless a scan holds several critical lesions. The count is averaged over the repeats of
    the cross-validation, in which a scan can go either way.
    Input:
        critical_lesions_csv: critical_lesions.csv written by evaluate_segmentation.py
        df_predictions: out-of-fold predictions of one auto run
    Output:
        A dict with the counts and the end-to-end sensitivity
    """
    df_critical = pd.read_csv(critical_lesions_csv)

    # Only the scans the classifier was evaluated on can be counted
    n_lesions_total = len(df_critical)
    df_critical = df_critical[df_critical["scan_id"].isin(set(df_predictions["scan_id"]))]
    if len(df_critical) < n_lesions_total:
        logger.warning(
            f"{n_lesions_total - len(df_critical)} critical lesion(s) are in scans the classifier was "
            "not evaluated on and are left out of the end-to-end sensitivity"
        )

    # Scans where the classifier flagged one of the critical predicted lesions as critical, counted
    # separately in every repeat and then averaged
    n_classified = 0.0
    for _, df_repeat in df_predictions.groupby("repeat"):
        recovered_scans = set(df_repeat.loc[(df_repeat["y_true"] == 1) & (df_repeat["y_pred"] == 1), "scan_id"])
        n_classified += int((df_critical["detected"] & df_critical["scan_id"].isin(recovered_scans)).sum())
    n_classified /= df_predictions["repeat"].nunique()

    return {
        "critical_lesions": len(df_critical),
        "detected_by_segmentation": int(df_critical["detected"].sum()),
        "detected_and_classified_critical": n_classified,
        "end_to_end_sensitivity": n_classified / len(df_critical) if len(df_critical) else np.nan,
    }


def plot_curves(df_predictions, output_png):
    """Pooled out-of-fold ROC, precision-recall and calibration curves of every run of the frame."""
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    for (arm, model), df_run in df_predictions.groupby(["arm", "model"], sort=False):
        if df_run["y_true"].nunique() < 2:
            continue
        name = f"{arm} / {model}"
        RocCurveDisplay.from_predictions(df_run["y_true"], df_run["y_prob"], name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(df_run["y_true"], df_run["y_prob"], name=name, ax=axes[1])
        CalibrationDisplay.from_predictions(df_run["y_true"], df_run["y_prob"], n_bins=10, strategy="quantile", name=name, ax=axes[2])
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="grey")
    for axis, title in zip(axes, ["ROC", "Precision-recall", "Calibration"]):
        axis.set_title(f"{title} (pooled out-of-fold)")
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close(figure)


def analyse_features(pipeline, X, output_folder, tag, n_display=15):
    """
    Rank the features of a fitted pipeline. Tree ensembles are ranked by mean absolute SHAP value,
    with the SHAP summary plots saved; linear models are ranked by the absolute value of their
    coefficients, which is why they are standardised. The ranking uses the transformed matrix, i.e.
    the one the classifier was fitted on.
    Output:
        The dataframe of features ranked by importance
    """
    classifier = pipeline.named_steps["clf"]
    # The classifier only ever saw the features the pipeline let through
    feature_names = X.columns
    if "selector" in pipeline.named_steps:
        feature_names = X.columns[pipeline.named_steps["selector"].get_support()]
    X_transformed = pd.DataFrame(pipeline[:-1].transform(X), columns=feature_names, index=X.index)

    if hasattr(classifier, "feature_importances_"):
        shap_values = shap.TreeExplainer(classifier)(X_transformed)
        # Random forests return one set of values per class, i.e. an array of shape
        # (lesions, features, classes): keep the critical class, which is the second one
        if shap_values.values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        importance, importance_type = np.abs(shap_values.values).mean(axis=0), "mean_abs_shap"
        for plot_function, name in [(shap.plots.beeswarm, "shap_beeswarm"), (shap.plots.bar, "shap_bar")]:
            plot_function(shap_values, max_display=n_display, show=False)
            plt.savefig(os.path.join(output_folder, f"{name}_{tag}.png"), bbox_inches="tight", dpi=200)
            plt.close("all")
    else:
        importance, importance_type = np.abs(classifier.coef_[0]), "abs_coefficient"

    df_importance = pd.DataFrame({
        "feature": feature_names, "importance": importance, "importance_type": importance_type,
    }).sort_values("importance", ascending=False, ignore_index=True)
    df_importance.to_csv(os.path.join(output_folder, f"top_features_{tag}.csv"), index=False)

    df_top = df_importance.head(n_display).iloc[::-1]
    plt.figure(figsize=(9, 7))
    plt.barh(df_top["feature"], df_top["importance"], color="darkseagreen")
    plt.xlabel(importance_type.replace("_", " "))
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"feature_importance_{tag}.png"), dpi=200)
    plt.close("all")

    return df_importance


def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    logger.add(os.path.join(args.output_folder, "train_classifier.log"), mode="w")
    # The command first, so that a result folder can be traced back to the run that produced it
    logger.info(f"Command: {sys.executable} {' '.join(shlex.quote(argument) for argument in sys.argv)}")
    logger.info(f"Run from: {os.getcwd()}")

    tables = {arm: pd.read_csv(os.path.join(args.features_folder, filename)) for arm, filename in ARM_TABLES.items()}

    # Keep the subjects present in both arms, so that the gap between the arms reflects the masks
    common_subjects = set.intersection(*(set(df["subject"]) for df in tables.values()))
    for arm, df in tables.items():
        dropped = sorted(set(df["subject"]) - common_subjects)
        if dropped:
            logger.warning(f"{arm}: {len(dropped)} subject(s) dropped, absent from the other arm: {dropped}")
        tables[arm] = df[df["subject"].isin(common_subjects)].reset_index(drop=True)

    feature_columns = get_feature_columns(tables, args.keep_norm_diff, args.features)
    counts = pd.DataFrame({
        arm: {"lesions": len(df), "critical": int(df["critical_lesion_label"].sum()), "subjects": df["subject"].nunique()}
        for arm, df in tables.items()
    }).T
    logger.info(f"Subjects shared by both arms: {len(common_subjects)} | features: {len(feature_columns)} ({args.features})")
    logger.info(f"Lesions, critical lesions and subjects of each arm:\n{counts.to_string()}")
    logger.info(f"Models compared: {', '.join(args.models)}")

    # The folds are sets of subjects, shared by every arm and every model
    folds = get_folds(tables["manual"], args.n_splits, args.n_repeats)
    pd.DataFrame([
        {"repeat": repeat, "fold": fold, "subject": subject}
        for repeat, fold, subjects in folds for subject in sorted(subjects)
    ]).to_csv(os.path.join(args.output_folder, "folds.csv"), index=False)

    # ------------------------------------------------- pass 1: compare the model families
    # Every family is run on all the features and, when asked, again with a selection fitted inside
    # each training fold. The two rows then sit side by side in the comparison table.
    selection_runs = [(None, "")]
    if args.select_k is not None:
        selection_runs.append((args.select_k, f" (select {args.select_k})"))

    fold_rows, predictions = [], []
    for model_name in args.models:
        for arm, df in tables.items():
            for select_k, suffix in selection_runs:
                rows, preds = run_nested_cv(
                    df, feature_columns, folds, MODEL_CONFIGS[model_name], arm, model_name + suffix,
                    args.n_splits_inner, args.n_iter, select_k,
                )
                fold_rows += rows
                predictions += preds

    df_folds = pd.DataFrame(fold_rows)
    logger.info(f"Lesion level, over {args.n_splits} subject-wise folds x {args.n_repeats} repeat(s):\n{format_summary(df_folds).to_string()}")

    # Best model of each arm, on which the second pass is run. Note that picking it on the same
    # cross-validation that scores it makes the winner's score optimistic.
    # Picked among the runs on all the features, since the second pass exists to rank every feature
    summary = summarise(df_folds)
    plain_runs = summary[summary.index.get_level_values("model").isin(args.models)]
    best_models = {arm: model for arm, model in plain_runs.groupby(level="arm")["f1_mean"].idxmax().values}
    logger.info(f"Best model of each arm, by mean F1: {best_models}")

    # ------- pass 2: final model of each arm, then the same nested CV on its top features alone
    for arm, model_name in best_models.items():
        logger.info(f"Fitting the final {arm} / {model_name} model on all its lesions...")
        X, y = get_xy(tables[arm], feature_columns)
        final_model, final_params = fit_model(MODEL_CONFIGS[model_name], X, y, tables[arm]["subject"], args.n_splits_inner, args.n_iter)
        logger.info(f"{arm} / {model_name}: best parameters of the final model {final_params}")
        df_importance = analyse_features(final_model, X, args.output_folder, f"{arm}_{model_name}")
        logger.info(f"{arm} / {model_name}: top {args.n_top_features} features\n{df_importance.head(args.n_top_features).to_string()}")

        top_features = df_importance["feature"].head(args.n_top_features).tolist()
        top_model_name = f"{model_name} (top {args.n_top_features})"
        rows, preds = run_nested_cv(
            tables[arm], top_features, folds, MODEL_CONFIGS[model_name], arm, top_model_name,
            args.n_splits_inner, args.n_iter,
        )
        fold_rows += rows
        predictions += preds

    # ----------------------------------------------------------------------------- reporting
    df_folds = pd.DataFrame(fold_rows)
    df_predictions = pd.concat(predictions, ignore_index=True)
    df_folds.to_csv(os.path.join(args.output_folder, "fold_results.csv"), index=False)
    df_predictions.to_csv(os.path.join(args.output_folder, "predictions.csv"), index=False)
    summarise(df_folds).to_csv(os.path.join(args.output_folder, "results_summary.csv"))
    logger.info(f"Lesion level, every run:\n{format_summary(df_folds).to_string()}")

    # Curves of every run, in two figures: a top-features run is not comparable to a full-feature one,
    # since its features were ranked with the labels of every subject
    is_top_features = df_predictions["model"].str.endswith(f"(top {args.n_top_features})")
    for df_subset, filename in [
        (df_predictions[~is_top_features], "roc_pr_calibration_all_features.png"),
        (df_predictions[is_top_features], f"roc_pr_calibration_top_{args.n_top_features}_features.png"),
    ]:
        if not df_subset.empty:
            plot_curves(df_subset, os.path.join(args.output_folder, filename))

    # Patient level: is the highest-scoring lesion of a patient one of its critical lesions. The
    # patients counted are those holding a critical lesion in the manual segmentation, in every run.
    critical_subjects = set(tables["manual"].loc[tables["manual"]["critical_lesion_label"] == 1, "subject"])
    df_patient = pd.DataFrame(
        {run: top_lesion_accuracy(df_run, critical_subjects)
         for run, df_run in df_predictions.groupby(["arm", "model"], sort=False)},
        index=["subjects_with_critical", "top_lesion_critical"],
    ).T
    df_patient["accuracy"] = df_patient["top_lesion_critical"] / df_patient["subjects_with_critical"]
    df_patient.index.names = ["arm", "model"]
    df_patient.to_csv(os.path.join(args.output_folder, "patient_level.csv"))
    logger.info(f"Patient level, is the highest-scoring lesion critical:\n{df_patient.round(3).to_string()}")

    # End to end: the critical lesions the segmentation missed count as errors of the whole pipeline
    if args.critical_lesions:
        df_end_to_end = pd.DataFrame.from_dict(
            {model: end_to_end_sensitivity(args.critical_lesions, df_run)
             for (arm, model), df_run in df_predictions.groupby(["arm", "model"], sort=False) if arm == "auto"},
            orient="index",
        )
        df_end_to_end.to_csv(os.path.join(args.output_folder, "end_to_end.csv"), index_label="model")
        logger.info(f"End to end, critical lesions of the manual segmentations:\n{df_end_to_end.round(3).to_string()}")

    logger.info(f"Analysis complete. Results saved to: {args.output_folder}")


if __name__ == "__main__":
    main()

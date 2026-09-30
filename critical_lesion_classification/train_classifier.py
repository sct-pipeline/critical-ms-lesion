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
    roc_pr_calibration.png        curves of the best model of each arm and of its top-features run
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
    --n-splits: number of outer cross-validation folds (default: 5)
    --n-splits-inner: number of inner folds of the hyperparameter search (default: 3)
    --n-iter: number of hyperparameter settings sampled by the Bayesian search (default: 20)
    --n-top-features: number of features kept in the second pass (default: 10)
    --keep-norm-diff: keep the (diff_with_hc_norm) features, which are dropped by default

Author: Pierre-Louis Benveniste
"""
import os
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


def parse_args():
    parser = argparse.ArgumentParser(description="Compare model families on the features of the predicted and of the manual lesion segmentations, with a subject-wise nested cross-validation.")
    parser.add_argument("-f", "--features-folder", type=str, required=True, help="Folder holding features_manual.csv and features_pred.csv (output of extract_features.py).")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder where the results will be saved.")
    parser.add_argument("--critical-lesions", type=str, help="Path to critical_lesions.csv (output of evaluate_segmentation.py). When given, the end-to-end sensitivity for critical lesions is reported, counting the lesions the segmentation missed as errors.")
    parser.add_argument("--models", type=str, nargs="+", default=list(MODEL_CONFIGS), choices=list(MODEL_CONFIGS), help="Model families to compare (default: all of them).")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of outer cross-validation folds (default: 5).")
    parser.add_argument("--n-splits-inner", type=int, default=3, help="Number of inner folds of the hyperparameter search (default: 3).")
    parser.add_argument("--n-iter", type=int, default=20, help="Number of hyperparameter settings sampled by the Bayesian search (default: 20).")
    parser.add_argument("--n-top-features", type=int, default=10, help="Number of features kept in the second pass (default: 10).")
    parser.add_argument("--keep-norm-diff", action="store_true", help="Keep the (diff_with_hc_norm) features, which are dropped by default.")
    return parser.parse_args()


def get_feature_columns(tables, keep_norm_diff):
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
    return columns


def get_xy(df, feature_columns):
    """
    Feature matrix and labels of a feature table. Infinite values (produced by the ratio-based
    asymmetry measures when a quadrant area is zero) become missing values, imputed by the pipeline.
    """
    X = df[feature_columns].replace([np.inf, -np.inf], np.nan).astype(float)
    return X, df["critical_lesion_label"].astype(int)


def get_folds(df_reference, n_splits):
    """
    Subject-wise cross-validation folds, stratified on the lesion label of the reference table. A
    fold is a set of subjects rather than row indices, so that every arm and every model is trained
    and evaluated on exactly the same patients. No subject is set aside: each one is evaluated once.
    Output:
        A list of sets of subjects, one per fold
    """
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    subjects = df_reference["subject"]
    return [
        set(subjects.iloc[evaluation_idx])
        for _, evaluation_idx in splitter.split(df_reference, df_reference["critical_lesion_label"], subjects)
    ]


def build_pipeline(config):
    """Mean imputation, standardisation for the models that need it, then the classifier."""
    steps = [("imputer", SimpleImputer(strategy="mean"))]
    if config.get("scale"):
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", clone(config["estimator"])))
    return Pipeline(steps)


def fit_model(config, X, y, groups, n_splits_inner, n_iter):
    """
    Fit one model family, its hyperparameters picked by a Bayesian search over an inner subject-wise
    cross-validation. Imputation and standardisation are inside the pipeline, hence fitted on the
    inner training folds only.
    Output:
        (best_estimator, best_params) with the estimator being the fitted pipeline
    """
    search = BayesSearchCV(
        build_pipeline(config), config["search_space"], cv=GroupKFold(n_splits_inner),
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


def run_nested_cv(df, feature_columns, folds, config, arm, model_name, n_splits_inner, n_iter):
    """
    Nested cross-validation of one model family on one arm: each fold in turn is the evaluation set,
    the model being fitted on the other folds with its hyperparameters chosen by a Bayesian search
    over an inner subject-wise cross-validation of those folds. Every subject is evaluated once and
    nothing is held out.
    Output:
        (fold_rows, predictions), the per-fold metrics and the out-of-fold predictions
    """
    fold_rows, predictions = [], []
    for fold, fold_subjects in enumerate(tqdm(folds, desc=f"{arm} / {model_name}"), start=1):
        is_evaluated = df["subject"].isin(fold_subjects)
        df_train, df_eval = df[~is_evaluated], df[is_evaluated]
        if df_eval.empty or df_train["critical_lesion_label"].nunique() < 2:
            logger.warning(f"{arm} / {model_name}, fold {fold}: skipped, nothing to evaluate or a single class in training")
            continue

        X_train, y_train = get_xy(df_train, feature_columns)
        model, best_params = fit_model(config, X_train, y_train, df_train["subject"], n_splits_inner, n_iter)
        logger.info(f"{arm} / {model_name}, fold {fold}: best parameters {best_params}")

        X_eval, y_eval = get_xy(df_eval, feature_columns)
        y_prob = model.predict_proba(X_eval)[:, 1]
        fold_rows.append({"arm": arm, "model": model_name, "fold": fold, **compute_metrics(y_eval, y_prob)})
        # matched only exists for the predicted lesions: it is left empty for the manual arm
        predictions.append(
            df_eval.reindex(columns=["subject", "scan_id", "lesion_label", "matched"])
            .assign(arm=arm, model=model_name, fold=fold, y_true=y_eval.values, y_prob=y_prob,
                    y_pred=(y_prob >= 0.5).astype(int))
        )

    return fold_rows, predictions


def summarise(df_folds):
    """
    Lesion counts summed and metrics averaged over the folds of every arm and model.
    Output:
        A dataframe indexed by (arm, model)
    """
    by_run = df_folds.groupby(["arm", "model"], sort=False)
    stats = by_run[METRICS].agg(["mean", "std"])
    stats.columns = [f"{metric}_{statistic}" for metric, statistic in stats.columns]
    return by_run[["n_lesions", "n_critical"]].sum().join(stats)


def format_summary(df_folds):
    """The same summary as a table of 'mean ± std' strings, for the log."""
    by_run = df_folds.groupby(["arm", "model"], sort=False)
    return by_run[["n_lesions"]].sum().join(by_run[METRICS].agg(lambda values: f"{values.mean():.3f} ± {values.std():.3f}"))


def top_lesion_accuracy(df_predictions, critical_subjects):
    """
    Patient-level check: among the patients holding at least one critical lesion in the manual
    segmentation, how often the lesion with the highest predicted probability is a critical one.
    The denominator is the same for every arm and model, so a patient whose critical lesion the
    segmentation missed counts as an error instead of dropping out of the comparison.
    Output:
        (n_subjects, n_correct)
    """
    df = df_predictions[df_predictions["subject"].isin(critical_subjects)]
    df_top = df.loc[df.groupby("subject")["y_prob"].idxmax()]
    return len(critical_subjects), int(df_top["y_true"].sum())


def end_to_end_sensitivity(critical_lesions_csv, df_predictions):
    """
    End-to-end sensitivity for critical lesions: of the critical lesions of the manual segmentations,
    how many were both found by the segmentation and classified critical. A lesion the segmentation
    missed counts as an error, since the classifier never gets to see it. A lesion that was found
    counts as recovered when a critical predicted lesion of its scan was classified critical, which
    is exact unless a scan holds several critical lesions.
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

    # Scans where the classifier flagged one of the critical predicted lesions as critical
    recovered_scans = set(df_predictions.loc[(df_predictions["y_true"] == 1) & (df_predictions["y_pred"] == 1), "scan_id"])
    n_classified = int((df_critical["detected"] & df_critical["scan_id"].isin(recovered_scans)).sum())

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
    X_transformed = pd.DataFrame(pipeline[:-1].transform(X), columns=X.columns, index=X.index)

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
        "feature": X.columns, "importance": importance, "importance_type": importance_type,
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

    tables = {arm: pd.read_csv(os.path.join(args.features_folder, filename)) for arm, filename in ARM_TABLES.items()}

    # Keep the subjects present in both arms, so that the gap between the arms reflects the masks
    common_subjects = set.intersection(*(set(df["subject"]) for df in tables.values()))
    for arm, df in tables.items():
        dropped = sorted(set(df["subject"]) - common_subjects)
        if dropped:
            logger.warning(f"{arm}: {len(dropped)} subject(s) dropped, absent from the other arm: {dropped}")
        tables[arm] = df[df["subject"].isin(common_subjects)].reset_index(drop=True)

    feature_columns = get_feature_columns(tables, args.keep_norm_diff)
    counts = pd.DataFrame({
        arm: {"lesions": len(df), "critical": int(df["critical_lesion_label"].sum()), "subjects": df["subject"].nunique()}
        for arm, df in tables.items()
    }).T
    logger.info(f"Subjects shared by both arms: {len(common_subjects)} | features: {len(feature_columns)}")
    logger.info(f"Lesions, critical lesions and subjects of each arm:\n{counts.to_string()}")
    logger.info(f"Models compared: {', '.join(args.models)}")

    # The folds are sets of subjects, shared by every arm and every model
    folds = get_folds(tables["manual"], args.n_splits)
    pd.DataFrame([
        {"subject": subject, "fold": fold} for fold, subjects in enumerate(folds, start=1) for subject in sorted(subjects)
    ]).to_csv(os.path.join(args.output_folder, "folds.csv"), index=False)

    # ------------------------------------------------- pass 1: compare the model families
    fold_rows, predictions = [], []
    for model_name in args.models:
        for arm, df in tables.items():
            rows, preds = run_nested_cv(
                df, feature_columns, folds, MODEL_CONFIGS[model_name], arm, model_name,
                args.n_splits_inner, args.n_iter,
            )
            fold_rows += rows
            predictions += preds

    df_folds = pd.DataFrame(fold_rows)
    logger.info(f"Lesion level, over {args.n_splits} subject-wise folds:\n{format_summary(df_folds).to_string()}")

    # Best model of each arm, on which the second pass is run. Note that picking it on the same
    # cross-validation that scores it makes the winner's score optimistic.
    summary = summarise(df_folds)
    best_models = {arm: model for arm, model in summary.groupby(level="arm")["f1_mean"].idxmax().values}
    logger.info(f"Best model of each arm, by mean F1: {best_models}")

    # ------- pass 2: final model of each arm, then the same nested CV on its top features alone
    plot_runs = []
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
        plot_runs += [(arm, model_name), (arm, top_model_name)]

    # ----------------------------------------------------------------------------- reporting
    df_folds = pd.DataFrame(fold_rows)
    df_predictions = pd.concat(predictions, ignore_index=True)
    df_folds.to_csv(os.path.join(args.output_folder, "fold_results.csv"), index=False)
    df_predictions.to_csv(os.path.join(args.output_folder, "predictions.csv"), index=False)
    summarise(df_folds).to_csv(os.path.join(args.output_folder, "results_summary.csv"))
    logger.info(f"Lesion level, every run:\n{format_summary(df_folds).to_string()}")

    # Curves of the best model of each arm and of its top-features run
    plot_curves(
        df_predictions.merge(pd.DataFrame(plot_runs, columns=["arm", "model"]), on=["arm", "model"]),
        os.path.join(args.output_folder, "roc_pr_calibration.png"),
    )

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

#!/usr/bin/env python
"""Run AUS vs US cross-country crop yield prediction experiments.

The pipeline is intentionally self-contained so it can run from the current
workspace with the local CSV and RAR archives.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import DtypeWarning, PerformanceWarning

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Lasso, Ridge, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor


warnings.filterwarnings("ignore", category=DtypeWarning)
warnings.filterwarnings("ignore", category=PerformanceWarning)

MONTHS = ["may", "jun", "jul", "aug", "sep", "oct"]
OVERLAP_CROPS = ["Barley", "Canola", "Oats", "Wheat"]
ID_COLS = ["country", "region", "crop", "year_start", "season"]
TARGET = "yield_t_ha"
CAT_COLS = ["country", "region", "crop"]
LEAKAGE_PATTERNS = [
    "yield",
    "production",
    "harvest",
    "area",
    "value",
    "unit_desc",
    "yield_unit",
    "short_desc",
    "statisticcat_desc",
    "commodity_desc",
    "class_desc",
    "source_desc",
    "sector_desc",
    "group_desc",
    "prodn_practice_desc",
    "util_practice_desc",
    "domain_desc",
    "cv_pct",
]


@dataclass(frozen=True)
class SplitSpec:
    name: str
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class FeatureSet:
    name: str
    numeric_cols: list[str]
    categorical_cols: list[str]


class FitStateRegressor(BaseEstimator, RegressorMixin):
    """Adapter for regressors that do not expose fitted state to sklearn Pipeline."""

    def __init__(self, estimator: Any):
        self.estimator = estimator

    def fit(self, X: Any, y: Any) -> "FitStateRegressor":
        self.estimator.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.estimator.predict(X))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run crop yield prediction experiments for AUS vs US paper."
    )
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument(
        "--out", type=Path, default=Path("outputs") / "paper_yield_study"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--train-year-cutoff", type=int, default=2015)
    parser.add_argument("--fair-end-year", type=int, default=2021)
    parser.add_argument("--catboost-iterations", type=int, default=300)
    parser.add_argument(
        "--skip-catboost",
        action="store_true",
        help="Skip CatBoost. By default CatBoost is trained on GPU.",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Allow CatBoost CPU fallback if GPU training fails.",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Skip optional US processed vs harmonized audit.",
    )
    return parser.parse_args()


def read_archive_csv(archive: Path, member: str) -> pd.DataFrame:
    if not archive.exists():
        raise FileNotFoundError(f"Missing archive: {archive}")
    tar_exe = shutil.which("tar")
    if tar_exe is None:
        raise RuntimeError("Windows tar.exe is required to stream the local RAR archives.")
    proc = subprocess.run(
        [tar_exe, "-xOf", str(archive), member],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    from io import BytesIO

    return pd.read_csv(BytesIO(proc.stdout), low_memory=False)


def load_datasets(data_root: Path) -> dict[str, pd.DataFrame]:
    aus_path = data_root / "train_ready_winter_crops_features_AUS.csv"
    if not aus_path.exists():
        raise FileNotFoundError(f"Missing Australia CSV: {aus_path}")
    datasets = {
        "aus": pd.read_csv(aus_path, low_memory=False),
        "us_harm_wheat": read_archive_csv(
            data_root / "harmonized_US.rar",
            "harmonized/us_harmonized_wheat_only.csv",
        ),
        "us_harm_overlap": read_archive_csv(
            data_root / "harmonized_US.rar",
            "harmonized/us_harmonized_all_overlap_crops.csv",
        ),
        "us_processed_wheat": read_archive_csv(
            data_root / "final_us_only.rar",
            "final_us_only/us_final_dataset_wheat_only.csv",
        ),
        "us_processed_all": read_archive_csv(
            data_root / "final_us_only.rar",
            "final_us_only/us_final_dataset_all_crops.csv",
        ),
    }
    for name, df in datasets.items():
        validate_required_columns(name, df)
        df["year_start"] = pd.to_numeric(df["year_start"], errors="coerce").astype("Int64")
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    return datasets


def validate_required_columns(name: str, df: pd.DataFrame) -> None:
    required = set(ID_COLS + [TARGET])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def fair_filter(df: pd.DataFrame, fair_end_year: int) -> pd.DataFrame:
    out = df.copy()
    out = out[out["year_start"].between(1989, fair_end_year)]
    out = out.dropna(subset=[TARGET, "year_start"])
    out["year_start"] = out["year_start"].astype(int)
    return out.reset_index(drop=True)


def add_or_get(df: pd.DataFrame, out_name: str, candidates: list[str]) -> None:
    for col in candidates:
        if col in df.columns:
            df[out_name] = pd.to_numeric(df[col], errors="coerce")
            return
    df[out_name] = np.nan


def add_sum(df: pd.DataFrame, out_name: str, candidates: list[str]) -> None:
    available = [c for c in candidates if c in df.columns]
    if available:
        df[out_name] = df[available].apply(pd.to_numeric, errors="coerce").sum(
            axis=1, min_count=1
        )
    else:
        df[out_name] = np.nan


def add_mean(df: pd.DataFrame, out_name: str, candidates: list[str]) -> None:
    available = [c for c in candidates if c in df.columns]
    if available:
        df[out_name] = df[available].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    else:
        df[out_name] = np.nan


def build_model_frame(raw_df: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    df = raw_df.copy()
    out = pd.DataFrame()
    for col in ID_COLS:
        out[col] = df[col]
    out[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    out["dataset_label"] = dataset_label

    for month in MONTHS:
        add_or_get(out, f"rain_mm_sum_{month}", [f"rain_mm_sum_{month}"])
        out[f"rain_mm_sum_{month}"] = pd.to_numeric(
            df.get(f"rain_mm_sum_{month}"), errors="coerce"
        )
        out[f"tmax_c_mean_{month}"] = pd.to_numeric(
            df.get(f"tmax_c_mean_{month}"), errors="coerce"
        )
        out[f"tmin_c_mean_{month}"] = pd.to_numeric(
            df.get(f"tmin_c_mean_{month}"), errors="coerce"
        )
        if f"tmean_c_mean_{month}" in df.columns:
            out[f"tmean_c_mean_{month}"] = pd.to_numeric(
                df[f"tmean_c_mean_{month}"], errors="coerce"
            )
        else:
            out[f"tmean_c_mean_{month}"] = (
                out[f"tmax_c_mean_{month}"] + out[f"tmin_c_mean_{month}"]
            ) / 2.0
        for threshold in [25, 30, 35]:
            out[f"heat_days_{threshold}_{month}"] = pd.to_numeric(
                df.get(f"heat_days_{threshold}_{month}"), errors="coerce"
            )
        out[f"dry_days_{month}"] = pd.to_numeric(
            df.get(f"dry_days_{month}"), errors="coerce"
        )
        if f"wet_days_{month}" in df.columns:
            out[f"wet_days_{month}"] = pd.to_numeric(
                df[f"wet_days_{month}"], errors="coerce"
            )
        else:
            out[f"wet_days_{month}"] = pd.to_numeric(
                df.get(f"rain_days_{month}"), errors="coerce"
            )
        out[f"radiation_mj_m2_sum_{month}"] = pd.to_numeric(
            df.get(f"radiation_mj_m2_sum_{month}"), errors="coerce"
        )

    add_or_get(out, "rain_may_oct", ["rain_may_oct"])
    if out["rain_may_oct"].isna().all():
        add_sum(out, "rain_may_oct", [f"rain_mm_sum_{m}" for m in MONTHS])

    add_or_get(out, "avg_tmax_may_oct", ["avg_tmax_may_oct"])
    if out["avg_tmax_may_oct"].isna().all():
        add_mean(out, "avg_tmax_may_oct", [f"tmax_c_mean_{m}" for m in MONTHS])

    add_or_get(out, "avg_tmin_may_oct", ["avg_tmin_may_oct"])
    if out["avg_tmin_may_oct"].isna().all():
        add_mean(out, "avg_tmin_may_oct", [f"tmin_c_mean_{m}" for m in MONTHS])

    add_mean(out, "avg_tmean_may_oct", [f"tmean_c_mean_{m}" for m in MONTHS])
    add_sum(out, "dry_days_may_oct", [f"dry_days_{m}" for m in MONTHS])
    add_sum(out, "wet_days_may_oct", [f"wet_days_{m}" for m in MONTHS])
    for threshold in [25, 30, 35]:
        add_sum(
            out,
            f"heat_days_{threshold}_may_oct",
            [f"heat_days_{threshold}_{m}" for m in MONTHS],
        )
    add_sum(out, "radiation_may_oct", [f"radiation_mj_m2_sum_{m}" for m in MONTHS])
    out["avg_radiation_may_oct"] = out["radiation_may_oct"] / 184.0

    add_soil_features(df, out)
    out = out.dropna(subset=[TARGET]).reset_index(drop=True)
    return out


def add_soil_features(src: pd.DataFrame, out: pd.DataFrame) -> None:
    depths = ["0_5", "5_15", "15_30", "30_60", "60_100", "100_200"]
    mappings = {
        "soil_clay": ["soil_clay", "soil_CLAY"],
        "soil_sand": ["soil_sand", "soil_SAND"],
        "soil_silt": ["soil_silt", "soil_SILT"],
        "soil_soc": ["soil_soc", "soil_SOC"],
        "soil_nitrogen": ["soil_nitrogen", "soil_TOTAL_N"],
        "soil_ph": ["soil_phh2o", "soil_PHC"],
        "soil_bulk_density": ["soil_bdod", "soil_BULK_DENSITY"],
        "soil_cec": ["soil_cec", "soil_ECEC"],
    }
    for concept, prefixes in mappings.items():
        for depth in depths:
            candidates = [f"{prefix}_{depth}" for prefix in prefixes]
            add_or_get_from_source(src, out, f"{concept}_{depth}", candidates)


def add_or_get_from_source(
    src: pd.DataFrame, out: pd.DataFrame, out_name: str, candidates: list[str]
) -> None:
    for col in candidates:
        if col in src.columns:
            out[out_name] = pd.to_numeric(src[col], errors="coerce")
            return
    out[out_name] = np.nan


def detect_leakage_columns(columns: list[str]) -> list[str]:
    leaks = []
    for col in columns:
        low = col.lower()
        if col in ID_COLS or col == TARGET:
            continue
        if any(pattern in low for pattern in LEAKAGE_PATTERNS):
            leaks.append(col)
    return sorted(set(leaks))


def make_feature_sets(model_frames: list[pd.DataFrame]) -> dict[str, FeatureSet]:
    all_cols = set(model_frames[0].columns)
    for frame in model_frames[1:]:
        all_cols &= set(frame.columns)

    numeric_candidates = sorted(
        c
        for c in all_cols
        if c not in set(ID_COLS + [TARGET, "dataset_label"])
        and pd.api.types.is_numeric_dtype(model_frames[0][c])
    )
    leaks = set(detect_leakage_columns(numeric_candidates))
    numeric_candidates = [c for c in numeric_candidates if c not in leaks]

    weather_cols = [
        c for c in numeric_candidates if not c.startswith("soil_") and c != "year_start"
    ]
    soil_cols = [c for c in numeric_candidates if c.startswith("soil_")]
    return {
        "weather_only": FeatureSet(
            "weather_only",
            numeric_cols=["year_start"] + weather_cols,
            categorical_cols=CAT_COLS,
        ),
        "weather_soil": FeatureSet(
            "weather_soil",
            numeric_cols=["year_start"] + weather_cols + soil_cols,
            categorical_cols=CAT_COLS,
        ),
    }


def make_preprocessor(feature_set: FeatureSet) -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_set.numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", encoder),
                    ]
                ),
                feature_set.categorical_cols,
            ),
        ],
        remainder="drop",
    )


def make_models(seed: int, args: argparse.Namespace) -> dict[str, Any]:
    try:
        bagging_tree = BaggingRegressor(
            estimator=DecisionTreeRegressor(max_depth=8, min_samples_leaf=3, random_state=seed),
            n_estimators=80,
            random_state=seed,
            n_jobs=-1,
        )
    except TypeError:
        bagging_tree = BaggingRegressor(
            base_estimator=DecisionTreeRegressor(
                max_depth=8, min_samples_leaf=3, random_state=seed
            ),
            n_estimators=80,
            random_state=seed,
            n_jobs=-1,
        )

    models: dict[str, Any] = {
        "baseline_mean": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0, random_state=seed),
        "lasso": Lasso(alpha=0.001, max_iter=20000, random_state=seed),
        "elastic_net": ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=20000, random_state=seed),
        "bayesian_ridge": BayesianRidge(),
        "huber": HuberRegressor(max_iter=1000),
        "sgd_huber": SGDRegressor(
            loss="huber",
            penalty="elasticnet",
            alpha=0.0001,
            l1_ratio=0.15,
            max_iter=5000,
            tol=1e-4,
            random_state=seed,
        ),
        "linear_svr": LinearSVR(C=1.0, epsilon=0.05, max_iter=20000, random_state=seed),
        "svr_rbf": SVR(C=10.0, epsilon=0.05, gamma="scale"),
        "kernel_ridge_rbf": KernelRidge(alpha=1.0, kernel="rbf", gamma=0.02),
        "knn_distance": KNeighborsRegressor(n_neighbors=7, weights="distance"),
        "decision_tree": DecisionTreeRegressor(
            max_depth=8,
            min_samples_leaf=3,
            random_state=seed,
        ),
        "extra_tree": ExtraTreeRegressor(
            max_depth=10,
            min_samples_leaf=3,
            random_state=seed,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=250,
            random_state=seed,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=300,
            random_state=seed,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
        "bagging_tree": bagging_tree,
        "adaboost_tree": AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=4, min_samples_leaf=4, random_state=seed),
            n_estimators=150,
            learning_rate=0.05,
            random_state=seed,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            random_state=seed,
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            random_state=seed,
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.01,
        ),
        "mlp": MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=0.001,
            learning_rate_init=0.001,
            early_stopping=True,
            max_iter=800,
            random_state=seed,
        ),
    }
    try:
        from xgboost import XGBRegressor

        models["xgboost"] = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        )
    except Exception as exc:
        print(f"XGBoost unavailable; skipping xgboost: {exc}")
    try:
        from lightgbm import LGBMRegressor

        models["lightgbm"] = LGBMRegressor(
            objective="regression",
            n_estimators=300,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception as exc:
        print(f"LightGBM unavailable; skipping lightgbm: {exc}")
    try:
        from ngboost import NGBRegressor

        models["ngboost"] = FitStateRegressor(
            NGBRegressor(
                n_estimators=250,
                learning_rate=0.03,
                random_state=seed,
                verbose=False,
            )
        )
    except Exception as exc:
        print(f"NGBoost unavailable; skipping ngboost: {exc}")
    if not args.skip_catboost:
        try:
            from catboost import CatBoostRegressor

            models["catboost_gpu"] = CatBoostRegressor(
                loss_function="RMSE",
                iterations=args.catboost_iterations,
                learning_rate=0.05,
                depth=6,
                random_seed=seed,
                verbose=False,
                task_type="GPU",
                devices="0",
                allow_writing_files=False,
            )
        except Exception as exc:
            if args.allow_cpu_fallback:
                print(f"CatBoost import failed; continuing without CatBoost: {exc}")
            else:
                raise RuntimeError(f"CatBoost import failed, GPU training unavailable: {exc}")
    return models


def get_feature_names(preprocessor: ColumnTransformer, feature_set: FeatureSet) -> list[str]:
    names: list[str] = []
    names.extend(feature_set.numeric_cols)
    cat_pipe = preprocessor.named_transformers_["cat"]
    encoder = cat_pipe.named_steps["onehot"]
    try:
        cat_names = list(encoder.get_feature_names_out(feature_set.categorical_cols))
    except AttributeError:
        cat_names = list(encoder.get_feature_names(feature_set.categorical_cols))
    names.extend(cat_names)
    return names


def train_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_set: FeatureSet,
    model_name: str,
    model: Any,
    args: argparse.Namespace,
) -> tuple[np.ndarray, Pipeline, str]:
    preprocessor = make_preprocessor(feature_set)
    estimator = model
    if model_name == "catboost_gpu":
        catboost_status = "gpu"
    else:
        catboost_status = "not_applicable"

    pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    X_train = train_df[feature_set.numeric_cols + feature_set.categorical_cols]
    y_train = train_df[TARGET]
    X_test = test_df[feature_set.numeric_cols + feature_set.categorical_cols]
    try:
        pipe.fit(X_train, y_train)
    except Exception as exc:
        if model_name == "catboost_gpu" and args.allow_cpu_fallback:
            from catboost import CatBoostRegressor

            cpu_model = CatBoostRegressor(
                loss_function="RMSE",
                iterations=args.catboost_iterations,
                learning_rate=0.05,
                depth=6,
                random_seed=args.seed,
                verbose=False,
                task_type="CPU",
                allow_writing_files=False,
            )
            pipe = Pipeline([("preprocessor", make_preprocessor(feature_set)), ("model", cpu_model)])
            pipe.fit(X_train, y_train)
            catboost_status = "cpu_fallback"
        elif model_name == "catboost_gpu":
            raise RuntimeError(
                "CatBoost GPU training failed. Re-run with --allow-cpu-fallback only "
                f"if CPU CatBoost is acceptable. Original error: {exc}"
            )
        else:
            raise
    predictions = pipe.predict(X_test)
    return predictions, pipe, catboost_status


def metric_row(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }


def random_split(df: pd.DataFrame, seed: int, test_size: float) -> SplitSpec:
    indices = np.arange(len(df))
    stratify = df["crop"] if df["crop"].nunique() > 1 and df["crop"].value_counts().min() >= 2 else None
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=stratify
    )
    return SplitSpec("random_80_20", np.array(train_idx), np.array(test_idx))


def time_split(df: pd.DataFrame, cutoff: int) -> SplitSpec:
    train_idx = df.index[df["year_start"] <= cutoff].to_numpy()
    test_idx = df.index[df["year_start"] > cutoff].to_numpy()
    return SplitSpec(f"time_train_le_{cutoff}_test_gt_{cutoff}", train_idx, test_idx)


def country_split(df: pd.DataFrame, train_country: str, test_country: str) -> SplitSpec:
    train_idx = df.index[df["country"] == train_country].to_numpy()
    test_idx = df.index[df["country"] == test_country].to_numpy()
    return SplitSpec(
        f"leave_country_out_train_{train_country}_test_{test_country}",
        train_idx,
        test_idx,
    )


def validate_split(df: pd.DataFrame, split: SplitSpec) -> None:
    if len(split.train_idx) == 0 or len(split.test_idx) == 0:
        raise ValueError(f"Empty train/test split for {split.name}")
    if df.iloc[split.test_idx][TARGET].isna().any():
        raise ValueError(f"Missing target in test split {split.name}")


def run_experiment(
    experiment: str,
    dataset_name: str,
    df: pd.DataFrame,
    splits: list[SplitSpec],
    feature_sets: dict[str, FeatureSet],
    models: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    errors: list[pd.DataFrame] = []
    importances: list[dict[str, Any]] = []
    for split in splits:
        validate_split(df, split)
        train_df = df.iloc[split.train_idx].reset_index(drop=True)
        test_df = df.iloc[split.test_idx].reset_index(drop=True)
        for feature_set in feature_sets.values():
            used_features = feature_set.numeric_cols + feature_set.categorical_cols
            leaks = detect_leakage_columns(used_features)
            if leaks:
                raise ValueError(f"Leakage columns selected in {experiment}: {leaks}")
            for model_name, model in models.items():
                y_pred, pipe, catboost_status = train_predict(
                    train_df, test_df, feature_set, model_name, model, args
                )
                row = {
                    "experiment": experiment,
                    "dataset": dataset_name,
                    "feature_set": feature_set.name,
                    "split": split.name,
                    "model": model_name,
                    "train_rows": len(train_df),
                    "test_rows": len(test_df),
                    "catboost_training": catboost_status,
                }
                row.update(metric_row(test_df[TARGET], y_pred))
                metrics.append(row)

                err = test_df[ID_COLS + [TARGET]].copy()
                err["experiment"] = experiment
                err["dataset"] = dataset_name
                err["feature_set"] = feature_set.name
                err["split"] = split.name
                err["model"] = model_name
                err["prediction"] = y_pred
                err["error"] = err["prediction"] - err[TARGET]
                err["abs_error"] = err["error"].abs()
                errors.append(err)

                importances.extend(
                    extract_importance(
                        experiment, dataset_name, split.name, feature_set, model_name, pipe
                    )
                )
    return metrics, errors, importances


def extract_importance(
    experiment: str,
    dataset_name: str,
    split_name: str,
    feature_set: FeatureSet,
    model_name: str,
    pipe: Pipeline,
) -> list[dict[str, Any]]:
    model = pipe.named_steps["model"]
    if not hasattr(model, "feature_importances_") and not hasattr(model, "coef_"):
        return []
    preprocessor = pipe.named_steps["preprocessor"]
    feature_names = get_feature_names(preprocessor, feature_set)
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    else:
        values = np.abs(np.asarray(model.coef_, dtype=float).ravel())
    if len(values) != len(feature_names):
        return []
    order = np.argsort(values)[::-1][:30]
    rows = []
    for rank, idx in enumerate(order, start=1):
        rows.append(
            {
                "experiment": experiment,
                "dataset": dataset_name,
                "split": split_name,
                "feature_set": feature_set.name,
                "model": model_name,
                "rank": rank,
                "feature": feature_names[idx],
                "importance": float(values[idx]),
            }
        )
    return rows


def eda_summary(datasets: dict[str, pd.DataFrame], fair_end_year: int) -> pd.DataFrame:
    rows = []
    for name, df in datasets.items():
        year = pd.to_numeric(df["year_start"], errors="coerce")
        target = pd.to_numeric(df[TARGET], errors="coerce")
        fair = df[year.between(1989, fair_end_year)]
        rows.append(
            {
                "dataset": name,
                "rows": len(df),
                "columns": len(df.columns),
                "fair_rows_1989_2021": len(fair),
                "target_missing": int(target.isna().sum()),
                "crops": ", ".join(sorted(map(str, df["crop"].dropna().unique()))),
                "year_min": int(year.min()),
                "year_max": int(year.max()),
                "yield_mean": float(target.mean()),
                "yield_std": float(target.std()),
                "yield_min": float(target.min()),
                "yield_max": float(target.max()),
            }
        )
    return pd.DataFrame(rows)


def write_eda_markdown(summary: pd.DataFrame, out_file: Path) -> None:
    lines = ["# EDA Summary", "", summary.to_markdown(index=False), ""]
    out_file.write_text("\n".join(lines), encoding="utf-8")


def write_feature_schema_report(
    out_file: Path,
    feature_sets: dict[str, FeatureSet],
    raw_datasets: dict[str, pd.DataFrame],
    catboost_available: bool,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Feature Schema Report",
        "",
        "## Leakage Policy",
        "",
        "The target is `yield_t_ha`. Yield originals, production, area, harvested-area, "
        "USDA metadata, unit/value fields, and any target-derived columns are excluded.",
        "",
        "## Feature Sets",
        "",
    ]
    for name, fs in feature_sets.items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Numeric features: {len(fs.numeric_cols)}",
                f"- Categorical features: {', '.join(fs.categorical_cols)}",
                f"- Columns: {', '.join(fs.numeric_cols + fs.categorical_cols)}",
                "",
            ]
        )
    lines.extend(["## Raw Leakage Candidates Dropped", ""])
    for name, df in raw_datasets.items():
        leaks = detect_leakage_columns(list(df.columns))
        lines.append(f"- {name}: {len(leaks)} candidate columns: {', '.join(leaks) if leaks else 'none'}")
    lines.extend(
        [
            "",
            "## Soil Mapping",
            "",
            "AUS uppercase soil columns and US SoilGrids-style columns are mapped to common "
            "concept names for clay, sand, silt, SOC, nitrogen, pH, bulk density, and CEC.",
            "",
            "## CatBoost GPU",
            "",
            f"- CatBoost requested: {not args.skip_catboost}",
            f"- CatBoost available in run: {catboost_available}",
            f"- GPU fallback allowed: {args.allow_cpu_fallback}",
        ]
    )
    out_file.write_text("\n".join(lines), encoding="utf-8")


def validate_metrics(metrics_df: pd.DataFrame) -> None:
    if metrics_df.empty:
        raise ValueError("No metrics produced.")
    required = ["experiment", "dataset", "feature_set", "split", "model", "MAE", "RMSE", "R2"]
    missing = sorted(set(required) - set(metrics_df.columns))
    if missing:
        raise ValueError(f"Metrics missing columns: {missing}")
    for col in ["MAE", "RMSE"]:
        if not np.isfinite(metrics_df[col]).all():
            raise ValueError(f"Non-finite values in metrics column {col}")
    if metrics_df["R2"].isna().any() or np.isinf(metrics_df["R2"]).any():
        raise ValueError("Non-finite values in R2.")


def make_figures(
    out_dir: Path,
    frames: dict[str, pd.DataFrame],
    metrics_df: pd.DataFrame,
    errors_df: pd.DataFrame,
    importance_df: pd.DataFrame,
) -> None:
    fig_dir = out_dir / "06_paper_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    combined = pd.concat(
        [frames["aus_overlap"], frames["us_overlap"]],
        ignore_index=True,
    )
    trend = (
        combined.groupby(["country", "crop", "year_start"], as_index=False)[TARGET]
        .mean()
        .sort_values("year_start")
    )
    plt.figure(figsize=(11, 6))
    for (country, crop), grp in trend.groupby(["country", "crop"]):
        plt.plot(grp["year_start"], grp[TARGET], label=f"{country} {crop}", alpha=0.8)
    plt.xlabel("Year")
    plt.ylabel("Yield t/ha")
    plt.title("Yield Trends by Country and Crop")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(fig_dir / "yield_trends.png", dpi=160)
    plt.close()

    best_rows = (
        metrics_df.sort_values(["experiment", "feature_set", "RMSE"])
        .groupby(["experiment", "feature_set"], as_index=False)
        .head(1)
    )
    labels = best_rows["experiment"] + "\n" + best_rows["feature_set"] + "\n" + best_rows["model"]
    plt.figure(figsize=(13, 6))
    plt.bar(np.arange(len(best_rows)), best_rows["RMSE"])
    plt.xticks(np.arange(len(best_rows)), labels, rotation=65, ha="right", fontsize=7)
    plt.ylabel("RMSE")
    plt.title("Best RMSE by Experiment and Feature Set")
    plt.tight_layout()
    plt.savefig(fig_dir / "metric_comparison_rmse.png", dpi=160)
    plt.close()

    best_metric = metrics_df.sort_values("RMSE").iloc[0]
    mask = (
        (errors_df["experiment"] == best_metric["experiment"])
        & (errors_df["feature_set"] == best_metric["feature_set"])
        & (errors_df["split"] == best_metric["split"])
        & (errors_df["model"] == best_metric["model"])
    )
    pred = errors_df[mask]
    plt.figure(figsize=(6, 6))
    plt.scatter(pred[TARGET], pred["prediction"], alpha=0.75)
    low = min(pred[TARGET].min(), pred["prediction"].min())
    high = max(pred[TARGET].max(), pred["prediction"].max())
    plt.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
    plt.xlabel("Actual yield t/ha")
    plt.ylabel("Predicted yield t/ha")
    plt.title("Predicted vs Actual for Best Overall Run")
    plt.tight_layout()
    plt.savefig(fig_dir / "predicted_vs_actual_best.png", dpi=160)
    plt.close()

    if not importance_df.empty:
        imp_source = importance_df.sort_values("importance", ascending=False).head(20)
        plt.figure(figsize=(9, 7))
        plt.barh(imp_source["feature"][::-1], imp_source["importance"][::-1])
        plt.xlabel("Importance")
        plt.title("Top Feature Importances")
        plt.tight_layout()
        plt.savefig(fig_dir / "feature_importance_top20.png", dpi=160)
        plt.close()


def write_paper_outline(out_file: Path, metrics_df: pd.DataFrame) -> None:
    best = (
        metrics_df.sort_values(["experiment", "feature_set", "RMSE"])
        .groupby(["experiment", "feature_set"], as_index=False)
        .head(1)
    )
    lines = [
        "# Final Paper Outline",
        "",
        "## Title",
        "",
        "Cross-country crop yield prediction using climate and soil features: Australia and United States comparison for wheat and overlapping field crops",
        "",
        "## Introduction",
        "",
        "Motivate climate-driven crop yield prediction and cross-country generalization.",
        "",
        "## Data",
        "",
        "Describe Australia winter crop data, US harmonized data, target `yield_t_ha`, fair 1989-2021 comparison window, and leakage controls.",
        "",
        "## Methods",
        "",
        "Describe weather-only and weather+soil feature sets, random/time/country transfer splits, baseline, Ridge, Random Forest, Gradient Boosting, and CatBoost GPU.",
        "",
        "## Results",
        "",
        "Best model per experiment and feature set:",
        "",
        best[
            ["experiment", "feature_set", "split", "model", "MAE", "RMSE", "R2"]
        ].to_markdown(index=False),
        "",
        "## Discussion",
        "",
        "Discuss transfer gap, domain shift, whether soil improves over weather-only, and whether wheat transfers more stably than multi-crop overlap.",
        "",
        "## Conclusion",
        "",
        "Summarize internal accuracy versus cross-country transfer behavior and future work on finer spatial data and management variables.",
        "",
    ]
    out_file.write_text("\n".join(lines), encoding="utf-8")


def write_error_analysis_markdown(out_file: Path, errors_df: pd.DataFrame) -> None:
    summary = (
        errors_df.groupby(["experiment", "feature_set", "model"], as_index=False)
        .agg(MAE=("abs_error", "mean"), bias=("error", "mean"), n=("error", "size"))
        .sort_values("MAE")
        .head(30)
    )
    lines = ["# Error Analysis", "", summary.to_markdown(index=False), ""]
    out_file.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_datasets(data_root)
    eda = eda_summary(raw, args.fair_end_year)
    eda.to_csv(out_dir / "01_eda_summary.csv", index=False)
    write_eda_markdown(eda, out_dir / "01_eda_summary.md")

    aus = fair_filter(raw["aus"], args.fair_end_year)
    us_wheat = fair_filter(raw["us_harm_wheat"], args.fair_end_year)
    us_overlap = fair_filter(raw["us_harm_overlap"], args.fair_end_year)
    us_processed_wheat = fair_filter(raw["us_processed_wheat"], args.fair_end_year)
    us_processed_all = fair_filter(raw["us_processed_all"], args.fair_end_year)

    frames = {
        "aus_wheat": build_model_frame(aus[aus["crop"] == "Wheat"], "aus_wheat"),
        "aus_overlap": build_model_frame(
            aus[aus["crop"].isin(OVERLAP_CROPS)], "aus_overlap"
        ),
        "us_wheat": build_model_frame(us_wheat, "us_wheat"),
        "us_overlap": build_model_frame(us_overlap, "us_overlap"),
        "us_processed_wheat": build_model_frame(us_processed_wheat, "us_processed_wheat"),
        "us_processed_overlap": build_model_frame(
            us_processed_all[us_processed_all["crop"].isin(OVERLAP_CROPS)],
            "us_processed_overlap",
        ),
    }
    feature_sets = make_feature_sets(
        [frames["aus_overlap"], frames["us_overlap"], frames["aus_wheat"], frames["us_wheat"]]
    )
    catboost_available = not args.skip_catboost
    write_feature_schema_report(
        out_dir / "02_feature_schema_report.md", feature_sets, raw, catboost_available, args
    )

    for name, frame in frames.items():
        if frame.empty:
            raise ValueError(f"Prepared frame is empty: {name}")
        if frame["year_start"].max() > args.fair_end_year:
            raise ValueError(f"{name} contains years beyond {args.fair_end_year}")

    models = make_models(args.seed, args)
    metrics_all: list[dict[str, Any]] = []
    errors_all: list[pd.DataFrame] = []
    importances_all: list[dict[str, Any]] = []

    experiment_specs: list[tuple[str, str, pd.DataFrame, list[SplitSpec]]] = [
        (
            "E1_AUS_Wheat_internal",
            "aus_wheat",
            frames["aus_wheat"],
            [
                random_split(frames["aus_wheat"], args.seed, args.test_size),
                time_split(frames["aus_wheat"], args.train_year_cutoff),
            ],
        ),
        (
            "E2_US_Wheat_internal",
            "us_wheat",
            frames["us_wheat"],
            [
                random_split(frames["us_wheat"], args.seed, args.test_size),
                time_split(frames["us_wheat"], args.train_year_cutoff),
            ],
        ),
        (
            "E4_AUS_overlap_internal",
            "aus_overlap",
            frames["aus_overlap"],
            [
                random_split(frames["aus_overlap"], args.seed, args.test_size),
                time_split(frames["aus_overlap"], args.train_year_cutoff),
            ],
        ),
        (
            "E5_US_overlap_internal",
            "us_overlap",
            frames["us_overlap"],
            [
                random_split(frames["us_overlap"], args.seed, args.test_size),
                time_split(frames["us_overlap"], args.train_year_cutoff),
            ],
        ),
    ]

    wheat_combined = pd.concat([frames["aus_wheat"], frames["us_wheat"]], ignore_index=True)
    overlap_combined = pd.concat(
        [frames["aus_overlap"], frames["us_overlap"]], ignore_index=True
    )
    experiment_specs.extend(
        [
            (
                "E3_Wheat_transfer",
                "aus_us_wheat",
                wheat_combined,
                [
                    country_split(wheat_combined, "Australia", "United States"),
                    country_split(wheat_combined, "United States", "Australia"),
                ],
            ),
            (
                "E6_Combined_overlap",
                "aus_us_overlap",
                overlap_combined,
                [
                    random_split(overlap_combined, args.seed, args.test_size),
                    time_split(overlap_combined, args.train_year_cutoff),
                    country_split(overlap_combined, "Australia", "United States"),
                    country_split(overlap_combined, "United States", "Australia"),
                ],
            ),
        ]
    )
    if not args.skip_audit:
        experiment_specs.extend(
            [
                (
                    "E7_US_harmonized_overlap_audit",
                    "us_overlap",
                    frames["us_overlap"],
                    [
                        random_split(frames["us_overlap"], args.seed, args.test_size),
                        time_split(frames["us_overlap"], args.train_year_cutoff),
                    ],
                ),
                (
                    "E7_US_processed_overlap_audit",
                    "us_processed_overlap",
                    frames["us_processed_overlap"],
                    [
                        random_split(frames["us_processed_overlap"], args.seed, args.test_size),
                        time_split(frames["us_processed_overlap"], args.train_year_cutoff),
                    ],
                ),
            ]
        )

    for experiment, dataset_name, frame, splits in experiment_specs:
        print(f"Running {experiment} ({dataset_name})...")
        m, e, imp = run_experiment(
            experiment, dataset_name, frame, splits, feature_sets, models, args
        )
        metrics_all.extend(m)
        errors_all.extend(e)
        importances_all.extend(imp)

    metrics_df = pd.DataFrame(metrics_all)
    validate_metrics(metrics_df)
    metrics_df.to_csv(out_dir / "03_model_metrics.csv", index=False)

    errors_df = pd.concat(errors_all, ignore_index=True)
    errors_df.to_csv(out_dir / "04_error_analysis.csv", index=False)
    write_error_analysis_markdown(out_dir / "04_error_analysis.md", errors_df)

    importance_df = pd.DataFrame(importances_all)
    if importance_df.empty:
        importance_df = pd.DataFrame(
            columns=[
                "experiment",
                "dataset",
                "split",
                "feature_set",
                "model",
                "rank",
                "feature",
                "importance",
            ]
        )
    importance_df.to_csv(out_dir / "05_feature_importance.csv", index=False)

    make_figures(out_dir, frames, metrics_df, errors_df, importance_df)
    write_paper_outline(out_dir / "07_final_paper_outline.md", metrics_df)

    expected = [
        "01_eda_summary.csv",
        "01_eda_summary.md",
        "02_feature_schema_report.md",
        "03_model_metrics.csv",
        "04_error_analysis.csv",
        "04_error_analysis.md",
        "05_feature_importance.csv",
        "07_final_paper_outline.md",
    ]
    missing_outputs = [name for name in expected if not (out_dir / name).exists()]
    if missing_outputs:
        raise ValueError(f"Missing expected outputs: {missing_outputs}")
    if not (out_dir / "06_paper_figures").exists():
        raise ValueError("Missing figure directory.")

    print(f"Done. Outputs written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

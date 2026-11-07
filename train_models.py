import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "credit_default.csv"
TARGET = "default_payment_next_month"


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    dataset = pd.read_csv(DATA_PATH)
    if TARGET not in dataset.columns:
        raise ValueError(f"Expected target column {TARGET!r} in {DATA_PATH.name}.")

    features = dataset.drop(columns=[TARGET, "ID"], errors="ignore")
    target = dataset[TARGET].astype(int)
    if len(features.columns) < 12 or len(dataset) < 500:
        raise ValueError("The dataset does not meet the assignment's minimum size.")
    if set(target.unique()) != {0, 1}:
        raise ValueError("This training script expects binary target values 0 and 1.")
    return features, target


def build_searches():
    return {
        "logistic_regression": (
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            {
                "logisticregression__C": [0.1, 1.0, 10.0],
                "logisticregression__class_weight": [None, "balanced"],
            },
        ),
        "decision_tree": (
            DecisionTreeClassifier(random_state=42),
            {
                "max_depth": [4, 8, None],
                "min_samples_leaf": [5, 20],
                "class_weight": [None, "balanced"],
            },
        ),
        "knn": (
            make_pipeline(StandardScaler(), KNeighborsClassifier()),
            {
                "kneighborsclassifier__n_neighbors": [5, 11, 21],
                "kneighborsclassifier__weights": ["uniform", "distance"],
            },
        ),
        "naive_bayes": (
            make_pipeline(StandardScaler(), GaussianNB()),
            {"gaussiannb__var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5]},
        ),
        "support_vector_machine": (
            make_pipeline(StandardScaler(), LinearSVC(max_iter=10000, random_state=42)),
            {
                "linearsvc__C": [0.01, 0.1, 1.0, 10.0],
                "linearsvc__class_weight": [None, "balanced"],
            },
        ),
    }


def calculate_metrics(model, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    predictions = model.predict(features)
    if hasattr(model, "predict_proba"):
        class_labels = list(model.classes_)
        scores = model.predict_proba(features)[:, class_labels.index(1)]
    else:
        scores = model.decision_function(features)

    return {
        "Accuracy": accuracy_score(target, predictions),
        "AUC": roc_auc_score(target, scores),
        "Precision": precision_score(target, predictions, zero_division=0),
        "Recall": recall_score(target, predictions, zero_division=0),
        "F1": f1_score(target, predictions, zero_division=0),
        "MCC": matthews_corrcoef(target, predictions),
    }


def main() -> None:
    features, target = load_dataset()
    features_train, features_test, target_train, target_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    model_dir = PROJECT_DIR / "model"
    model_dir.mkdir(exist_ok=True)
    test_data = features_test.copy()
    test_data[TARGET] = target_test
    test_data.to_csv(PROJECT_DIR / "test_data.csv", index=False)

    results = []
    tuning_results = {}
    cross_validation = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for model_name, (estimator, parameter_grid) in build_searches().items():
        print(f"Tuning {model_name.replace('_', ' ')}...")
        search = GridSearchCV(
            estimator=estimator,
            param_grid=parameter_grid,
            scoring="f1",
            cv=cross_validation,
            n_jobs=-1,
            pre_dispatch="n_jobs",
        )
        search.fit(features_train, target_train)
        model = search.best_estimator_
        metrics = calculate_metrics(model, features_test, target_test)
        joblib.dump(model, model_dir / f"{model_name}.joblib")
        results.append({"Model": model_name.replace("_", " ").title(), **metrics})
        tuning_results[model_name] = {
            "best_cv_f1": float(search.best_score_),
            "best_params": search.best_params_,
        }
        print(f"Best CV F1: {search.best_score_:.4f}; parameters: {search.best_params_}")

    results_frame = pd.DataFrame(results).set_index("Model")
    results_frame.to_csv(PROJECT_DIR / "model_metrics.csv", float_format="%.6f")
    (PROJECT_DIR / "model_tuning.json").write_text(
        json.dumps(tuning_results, indent=2),
        encoding="utf-8",
    )

    print(f"Training rows: {len(features_train)}")
    print(f"Test rows: {len(features_test)}")
    print(results_frame.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"Saved five models to {model_dir}")
    print(f"Saved labeled test data to {PROJECT_DIR / 'test_data.csv'}")


if __name__ == "__main__":
    main()
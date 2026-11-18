from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "iris_model.joblib"


def main() -> None:
    iris = load_iris()
    features_train, features_test, labels_train, labels_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=200),
    )
    model.fit(features_train, labels_train)

    predictions = model.predict(features_test)
    print(f"Accuracy: {accuracy_score(labels_test, predictions):.2f}")
    print(
        classification_report(
            labels_test,
            predictions,
            target_names=iris.target_names,
        )
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "target_names": iris.target_names},
        MODEL_PATH,
    )
    print(f"Saved trained model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()

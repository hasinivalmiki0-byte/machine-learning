import argparse
from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "iris_model.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict an Iris species from four measurements (in cm)."
    )
    parser.add_argument("measurements", nargs=4, type=float, metavar="CM")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        parser.error("Trained model not found. Run `python src/train.py` first.")

    saved_data = joblib.load(MODEL_PATH)
    model = saved_data["model"]
    target_names = saved_data["target_names"]
    prediction = model.predict([args.measurements])[0]
    print(f"Predicted species: {target_names[prediction]}")


if __name__ == "__main__":
    main()

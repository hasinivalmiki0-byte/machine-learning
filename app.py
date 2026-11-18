from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import joblib


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "iris_model.joblib"
FEATURES = (
    "Sepal length (cm)",
    "Sepal width (cm)",
    "Petal length (cm)",
    "Petal width (cm)",
)


def main() -> None:
    root = tk.Tk()
    root.title("Iris Flower Classifier")
    root.resizable(False, False)

    if not MODEL_PATH.exists():
        messagebox.showerror(
            "Model not found",
            "Train the model first by running:\npython src/train.py",
            parent=root,
        )
        root.destroy()
        return

    saved_data = joblib.load(MODEL_PATH)
    model = saved_data["model"]
    target_names = saved_data["target_names"]

    frame = ttk.Frame(root, padding=20)
    frame.grid()
    ttk.Label(
        frame,
        text="Enter the flower measurements in centimeters:",
    ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

    entries: list[ttk.Entry] = []
    for row, feature in enumerate(FEATURES, start=1):
        ttk.Label(frame, text=feature).grid(
            row=row,
            column=0,
            sticky="w",
            padx=(0, 12),
            pady=4,
        )
        entry = ttk.Entry(frame, width=18)
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        entries.append(entry)

    result = ttk.Label(frame, text="Prediction will appear here.")
    result.grid(row=6, column=0, columnspan=2, sticky="w", pady=(12, 0))

    def predict() -> None:
        try:
            measurements = [float(entry.get()) for entry in entries]
        except ValueError:
            messagebox.showerror(
                "Invalid measurements",
                "Enter a number for each of the four measurements.",
                parent=root,
            )
            return

        if any(value <= 0 for value in measurements):
            messagebox.showerror(
                "Invalid measurements",
                "Measurements must be greater than zero.",
                parent=root,
            )
            return

        prediction = model.predict([measurements])[0]
        result.config(text=f"Predicted species: {target_names[prediction]}")

    ttk.Button(frame, text="Predict species", command=predict).grid(
        row=5,
        column=0,
        columnspan=2,
        sticky="ew",
        pady=(12, 0),
    )
    root.mainloop()


if __name__ == "__main__":
    main()

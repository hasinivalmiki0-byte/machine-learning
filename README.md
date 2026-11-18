# Machine Learning: Iris Flower Classifier

A beginner-friendly machine-learning project that predicts an Iris flower
species from its sepal and petal measurements.

## Setup

From the project folder, install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Train the model

```powershell
python src/train.py
```

The script loads the built-in Iris dataset, trains a logistic regression
pipeline, prints evaluation results, and saves the trained model to
`models/iris_model.joblib`.

## Make a prediction

Provide sepal length, sepal width, petal length, and petal width, in
centimeters:

```powershell
python src/predict.py 5.1 3.5 1.4 0.2
```

Train the model first before making a prediction.

## Use the input form

After training the model, open a desktop form to enter the measurements:

```powershell
python src/app.py
```

Enter all four measurements in centimeters and click **Predict species**.

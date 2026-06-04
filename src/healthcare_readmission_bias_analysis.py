"""Privacy-Preserving Healthcare Readmission Prediction

This script cleans the UCI diabetic readmission dataset, balances the target class,
trains a differentially private Random Forest model, and saves evaluation plots.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from diffprivlib.models import RandomForestClassifier as DP_RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "diabetic_readmission_data.csv"
OUTPUT_DIR = BASE_DIR / "screenshots"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["encounter_id", "patient_nbr", "weight", "payer_code", "medical_specialty"], errors="ignore")
    df = df[(df["race"] != "?") & (df["gender"] != "Unknown/Invalid")].copy()
    df["readmitted"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)

    label_cols = df.select_dtypes(include="object").columns
    encoders = {}
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    readmitted_1 = df[df["readmitted"] == 1]
    readmitted_0 = df[df["readmitted"] == 0].sample(n=len(readmitted_1), random_state=42)
    df_balanced = pd.concat([readmitted_1, readmitted_0]).sample(frac=1, random_state=42)

    X = df_balanced.drop(columns=["readmitted"])
    y = df_balanced["readmitted"]
    return df, X, y, encoders


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    model = DP_RandomForestClassifier(n_estimators=50, epsilon=1.0, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_test, y_test, y_pred


def save_prediction_distribution(y_pred):
    pred_counts = np.bincount(y_pred)
    labels = ["Not Readmitted", "Readmitted"]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, pred_counts)
    plt.title("Model Predictions: Readmission")
    plt.ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "prediction_distribution.png", dpi=160)
    plt.close()


def save_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    labels = ["True Negatives", "False Positives", "False Negatives", "True Positives"]
    values = [tn, fp, fn, tp]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.title("Confusion Matrix Breakdown")
    for i, val in enumerate(values):
        plt.text(i, val + 1, str(val), ha="center")
    plt.ylabel("Number of Cases")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix_breakdown.png", dpi=160)
    plt.close()


def save_predictions_by_race(df, X_test, y_pred, encoders):
    race_labels = df.loc[X_test.index, "race"]
    decoded_race = encoders["race"].inverse_transform(race_labels)
    viz_df = pd.DataFrame({"race": decoded_race, "prediction": y_pred})
    race_pred_counts = viz_df.groupby(["race", "prediction"]).size().unstack(fill_value=0)
    race_pred_counts.plot(kind="bar", stacked=True, figsize=(8, 5))
    plt.title("Predicted Readmissions by Race")
    plt.ylabel("Number of Patients")
    plt.xlabel("Race")
    plt.legend(["Not Readmitted", "Readmitted"])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predictions_by_race.png", dpi=160)
    plt.close()


def main():
    df, X, y, encoders = load_and_preprocess_data()
    model, X_test, y_test, y_pred = train_model(X, y)
    print("Classification Report:
", classification_report(y_test, y_pred))
    save_prediction_distribution(y_pred)
    save_confusion_matrix(y_test, y_pred)
    save_predictions_by_race(df, X_test, y_pred, encoders)
    print(f"Plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

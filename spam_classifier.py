"""Spam Email Detection Pipeline - TF-IDF + Logistic Regression."""
import argparse
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


def load_data(csv_path: str = "mail_data.csv") -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    df = raw.where(pd.notnull(raw), "")
    df = df.copy()
    df["label"] = df["Category"].map({"spam": 0, "ham": 1}).astype(int)
    return df


def train_and_evaluate(csv_path: str = "mail_data.csv", save_cm: bool = True):
    df = load_data(csv_path)
    X_train, X_test, Y_train, Y_test = train_test_split(
        df["Message"], df["label"], test_size=0.2, random_state=3
    )

    vectorizer = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_features, Y_train)

    train_pred = model.predict(X_train_features)
    test_pred = model.predict(X_test_features)

    metrics = {
        "rows": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_accuracy": accuracy_score(Y_train, train_pred),
        "test_accuracy": accuracy_score(Y_test, test_pred),
        "spam_precision": precision_score(Y_test, test_pred, pos_label=0),
        "spam_recall": recall_score(Y_test, test_pred, pos_label=0),
        "spam_f1": f1_score(Y_test, test_pred, pos_label=0),
        "ham_precision": precision_score(Y_test, test_pred, pos_label=1),
        "ham_recall": recall_score(Y_test, test_pred, pos_label=1),
        "ham_f1": f1_score(Y_test, test_pred, pos_label=1),
        "confusion_matrix": confusion_matrix(Y_test, test_pred, labels=[0, 1]),
        "report": classification_report(
            Y_test, test_pred, labels=[0, 1], target_names=["Spam (0)", "Ham (1)"]
        ),
    }

    if save_cm:
        os.makedirs("assets", exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4.8))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=metrics["confusion_matrix"],
            display_labels=["Spam (0)", "Ham (1)"],
        )
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Spam Email Classifier — Confusion Matrix (Test n=1,115)")
        plt.tight_layout()
        fig.savefig("assets/confusion_matrix.png", dpi=150)
        plt.close(fig)

    return model, vectorizer, metrics


def predict_message(model, vectorizer, text: str) -> str:
    features = vectorizer.transform([text])
    pred = model.predict(features)[0]
    return "Ham (Not Spam)" if pred == 1 else "Spam"


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate or predict spam messages.")
    parser.add_argument("--data", default="mail_data.csv", help="Path to CSV dataset.")
    parser.add_argument("--text", help="Optional message text to classify.")
    parser.add_argument("--no-plot", action="store_true", help="Skip saving confusion matrix plot.")
    args = parser.parse_args()

    model, vectorizer, m = train_and_evaluate(args.data, save_cm=not args.no_plot)
    print(f"Dataset rows: {m['rows']} (Train: {m['train_rows']}, Test: {m['test_rows']})")
    print(f"Train Accuracy: {m['train_accuracy']:.6f} ({m['train_accuracy']*100:.2f}%)")
    print(f"Test Accuracy:  {m['test_accuracy']:.6f} ({m['test_accuracy']*100:.2f}%)")
    print("\nClassification Report:\n" + m["report"])
    print("Confusion Matrix [[TN_spam, FP], [FN, TP_ham]]:")
    print(m["confusion_matrix"])

    if args.text:
        print(f"\nInput: {args.text}\nPrediction: {predict_message(model, vectorizer, args.text)}")


if __name__ == "__main__":
    main()

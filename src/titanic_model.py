"""
Titanic Survival Prediction
Beginner-friendly ML project
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def clean_data(df):
    # Select useful columns
    df = df[
        [
            "Survived",
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked",
        ]
    ].copy()

    # Fill missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Convert text columns into numbers
    sex_encoder = LabelEncoder()
    df["Sex"] = sex_encoder.fit_transform(df["Sex"])

    embarked_encoder = LabelEncoder()
    df["Embarked"] = embarked_encoder.fit_transform(df["Embarked"])

    return df


def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    best_model_name = None
    best_accuracy = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        print("\n==============================")
        print(f"Model: {name}")
        print(f"Accuracy: {accuracy:.4f}")
        print("==============================")
        print(classification_report(y_test, predictions))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name

    print("\nBest Model:")
    print(f"{best_model_name} with accuracy {best_accuracy:.4f}")


def main():
    file_path = "data/train.csv"

    print("Loading dataset...")
    df = load_data(file_path)

    print("Original data shape:", df.shape)

    print("Cleaning data...")
    df = clean_data(df)

    print("Cleaned data shape:", df.shape)

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("Training models...")
    train_models(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()

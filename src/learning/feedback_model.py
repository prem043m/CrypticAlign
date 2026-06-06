from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.model_selection import (
    train_test_split
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

import pandas as pd

class FeedbackModel:

    def __init__(self):

        self.model = LogisticRegression(
            class_weight="balanced",
            random_state=42,
            max_iter=1000
        )

    def train(
        self,
        dataset
    ):

        X = dataset.drop(
            columns=["label"]
        )

        y = dataset["label"]

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )
        )

        self.model.fit(
            X_train,
            y_train
        )

        predictions = (
            self.model.predict(X_test)
        )

        probabilities = (
            self.model.predict_proba(X_test)[:, 1]
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        try:
            roc_auc = roc_auc_score(
                y_test,
                probabilities
            )
            roc_auc_str = f"{roc_auc:.4f}"
        except Exception:
            roc_auc_str = "N/A"

        print("Model Evaluation Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  ROC AUC:   {roc_auc_str}")
        print()

        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        print("Confusion Matrix:")
        print(f"  TP: {tp} | FN: {fn}")
        print(f"  FP: {fp} | TN: {tn}")
        print()

        return accuracy

    def predict_probability(
        self,
        features
    ):
        feature_df = pd.DataFrame(
            [features],
            columns=[
                "text_similarity",
                "mbti_score",
                "profession_score",
                "career_goal_score",
                "location_score",
                "experience_score",
                "skills_score",
                "networking_intent_score"
            ]
        )
        return (
            self.model.predict_proba(
                feature_df
            )[0][1]
        )